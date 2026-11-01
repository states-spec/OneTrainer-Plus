import os
import shutil
import subprocess
import sys
import threading
import webbrowser


def open_url(url: str):
    """
    Opens url in the desktop's web browser, as webbrowser.open(url, new=0, autoraise=False) does.

    On a Linux desktop, webbrowser.open starts xdg-open with OneTrainer's terminal as its output, and the browser that
    xdg-open starts keeps it: the Flatpak Firefox, for example, printed "[2] Sandbox: CanCreateUserNamespace() clone()
    failure: EPERM" into OneTrainer's terminal at every click on a link (harmless: Flatpak doesn't let Firefox's own
    sandbox create a user namespace, and 2 is Firefox's PID inside Flatpak's). So xdg-open runs here with its output
    discarded, as webbrowser already does when it starts Firefox or Chrome itself. A BROWSER setting, other platforms
    and URLs other than http(s) still go to webbrowser.open.
    """
    xdg_open = _xdg_open()
    if xdg_open is None or not url.startswith(("http://", "https://")):
        webbrowser.open(url, new=0, autoraise=False)
        return

    try:
        process = subprocess.Popen(
            [xdg_open, url],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            close_fds=True,
            start_new_session=True,
        )
    except OSError:
        webbrowser.open(url, new=0, autoraise=False)
        return

    def wait():
        # xdg-open exits once the browser has the URL. Its own error message is discarded with the rest, so a failure
        # names the URL here
        return_code = process.wait()
        if return_code != 0:
            print(f"Could not open {url} in a web browser (xdg-open exited with status {return_code})")

    threading.Thread(target=wait, name="xdg-open", daemon=True).start()


def _xdg_open() -> str | None:
    # where webbrowser would use xdg-open (or the default browser's own command): a Linux desktop (an X11 or Wayland
    # display) without a BROWSER setting
    if sys.platform != "linux" or "BROWSER" in os.environ:
        return None
    if not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
        return None
    return shutil.which("xdg-open")
