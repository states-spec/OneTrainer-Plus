"""Which SSH server keys a cloud run accepts.

Fabric accepts any server key (paramiko's AutoAddPolicy) and loads no known_hosts, and scp ran with
StrictHostKeyChecking=no: anyone in the network path could pose as the cloud machine and get the SSH password, the
Hugging Face token and the dataset, and then answer the cloud trainer's file list and callbacks. Now:

- A Linux cloud is a server you know, so its key has to be in ~/.ssh/known_hosts. An unknown key stops the run with
  the key's fingerprint and the ssh command that adds it; a changed key stops it too (paramiko's own check).
- A RunPod pod gets a new key whenever it is created, so there is nothing to check its first key against. Its key is
  pinned once the run's first connection has authenticated, and every later connection of the run (callbacks,
  commands, file sync, reconnects, scp) has to present the same key.
"""
import base64
import hashlib
import os
import shutil
import tempfile
import threading

import fabric
import paramiko


def host_key_name(host: str, port: int) -> str:
    # how paramiko and OpenSSH name a server in known_hosts
    return host if port == 22 else f"[{host}]:{port}"


def fingerprint(key: paramiko.PKey) -> str:
    # as ssh shows it
    return "SHA256:" + base64.b64encode(hashlib.sha256(key.asbytes()).digest()).decode("ascii").rstrip("=")


class _NotInKnownHosts(paramiko.MissingHostKeyPolicy):
    def __init__(self, host: str, port: int, user: str):
        self.host = host
        self.port = port
        self.user = user

    def missing_host_key(self, client, hostname, key):
        raise paramiko.SSHException(
            f"The cloud machine {hostname} is not in ~/.ssh/known_hosts, so its identity can't be checked. Its "
            f"{key.get_name()} key has the fingerprint {fingerprint(key)}: compare it with your provider's if you "
            f"can, then add the key by connecting once in a terminal: ssh -p {self.port} {self.user}@{self.host}")


class _PinnedForThisRun(paramiko.MissingHostKeyPolicy):
    def __init__(self, host_keys: "HostKeys"):
        self.host_keys = host_keys

    def missing_host_key(self, client, hostname, key):
        self.host_keys.check_pinned(hostname, key)


class HostKeys:
    def __init__(self, trust_first_key: bool):
        self.trust_first_key = trust_first_key
        self.__pinned: dict[str, paramiko.PKey] = {}
        self.__lock = threading.Lock()
        self.__temp_dir: str | None = None

    def connection(self, host: str, port, user: str, connect_kwargs: dict) -> fabric.Connection:
        port = int(port)  # a port given as text ("22") made paramiko look up "[host]:22" instead of "host"
        connection = fabric.Connection(host=host, port=port, user=user, connect_kwargs=connect_kwargs)
        if self.trust_first_key:
            connection.client.set_missing_host_key_policy(_PinnedForThisRun(self))
        else:
            connection.client.load_system_host_keys()
            connection.client.set_missing_host_key_policy(_NotInKnownHosts(host, port, user))
        return connection

    def check_pinned(self, hostname: str, key: paramiko.PKey):
        with self.__lock:
            pinned = self.__pinned.get(hostname)
        # before the pin (the run's first connection) any key is accepted; pin() records it once authenticated
        if pinned is not None and (pinned.get_name() != key.get_name() or pinned.asbytes() != key.asbytes()):
            raise paramiko.BadHostKeyException(hostname, key, pinned)

    def pin(self, connection: fabric.Connection):
        """Pins the key of an open, authenticated connection for the rest of the run (RunPod only)."""
        if not self.trust_first_key:
            return
        key = connection.client.get_transport().get_remote_server_key()
        hostname = host_key_name(connection.host, int(connection.port))
        with self.__lock:
            pinned = self.__pinned.setdefault(hostname, key)
        if pinned is key:
            print(f"Cloud: {hostname} identified itself with the {key.get_name()} key {fingerprint(key)}; "
                  f"this run accepts only that key from now on")
        else:
            self.check_pinned(hostname, key)

    def scp_options(self) -> list[str]:
        """ssh options for scp that check the server key the same way."""
        if not self.trust_first_key:
            return ["-o", "StrictHostKeyChecking=yes"]
        with self.__lock:
            if self.__temp_dir is None:
                self.__temp_dir = tempfile.mkdtemp(prefix="onetrainer_known_hosts_")
            known_hosts = paramiko.HostKeys()
            for hostname, key in self.__pinned.items():
                known_hosts.add(hostname, key.get_name(), key)
            path = os.path.join(self.__temp_dir, "known_hosts")
            known_hosts.save(path)
        return ["-o", "StrictHostKeyChecking=yes", "-o", f"UserKnownHostsFile={path}"]

    def close(self):
        with self.__lock:
            if self.__temp_dir is not None:
                shutil.rmtree(self.__temp_dir, ignore_errors=True)
                self.__temp_dir = None
