[Russian version](how-to.ru.md)

Docker with hardware OpenGL through VNC
=======================================

In `hexlib`, there is now a portable graphical shell that works
on Windows and Linux; it uses OpenGL.

My goal was to run it on Linux on a remote machine in a Docker container with hardware
OpenGL support and access via VNC.

Surprisingly, this turned out to be not so simple.

Launching X-server and OpenGL support
-------------------------------------

To run a program on OpenGL, you need a running X-server. It uses a special
extension for OpenGL support. On desktops, this is called GLX; there's also
some EGL for embedded systems. I tried to get hardware support for GLX.

I inherited my image from the standard `nvidia/cudagl:X.X.X` image.

One of the difficulties was launching the X-server with hardware OpenGL support.
On the internet, people usually advise running the X-server on the host, and inside Docker, just
forward access to the socket, which is the `/tmp/.X11-unix` directory.

But I wanted to run everything inside the Docker image, and this became a real problem.

If the X-server is on the host, everything is fine, and there is hardware support.

However, if it's inside the container, the X-server starts, everything is good, but there is no hardware
support for OpenGL. You can check, for example, with the utility `glxinfo | grep vendor`,
which shows the vendor as MESA instead of NVIDIA. MESA is some kind of software emulation on the CPU.

The only thing that helped was a strange action: installing the NVIDIA driver
inside the container. There's an example in the [Dockerfile](./example.docker).
It's very strange that NVIDIA hasn't already done everything needed in their container.
This installation produces errors because it cannot overwrite some files.
Nevertheless, after this action, there are many more NVIDIA .so files,
and as a result, hardware GLX is available.

Creating xorg.conf configuration
---------------------------------

If there is no physical monitor on the machine, to launch the X-server,
you need to simulate a monitor by creating a special `xorg.conf` configuration.

Example of creating a configuration, with a virtual display:

```
sudo nvidia-xconfig -a \
    --allow-empty-initial-configuration \
    --use-display-device=None \
    --virtual=1920x1080 \
    --output-xconfig=xorg.conf
```

The option with a fake EDID also works:

```
sudo nvidia-xconfig -a \
    --connected-monitor=DFP-0 \
    --custom-edid=DFP-0:/etc/X11/fake_edid.hex \
    --mode-list=1920x1080 \
    --output-xconfig=xorg.conf
```

I tried different methods, but I didn't notice any difference in terms of
hardware support.

To have hardware GLX when launching the X-server inside the container, the only thing that helps is
installing the NVIDIA driver inside the container (even if it reports errors).

VNC and displays
-----------------

It turns out there are two ways to work with VNC.

### Method 1

Run everything on a single display, including Xorg, desktop, and VNC. Apparently, the VNC server
then grabs the screen, here's a quote:

> x0vncserver makes any X display available remotely. Unlike Xvnc,
it doesn't create a virtual display. Instead, it simply provides access
to an existing X server (usually the one connected to a physical screen).
XDamage will be used if the existing X server supports it;
otherwise, x0vncserver will resort to polling the screen to detect changes.

In the case of the TurboVNC server, I couldn't get it to run like this.

In the case of the TigerVNC server, I managed to do it. But I didn't like the result,
as it seemed to update more slowly than with method 2, and it didn't support changing
the screen size upon request from the VNC client.

### Method 2

The VNC server creates its own X-server for VNC.
Apparently, the VNC server then intercepts the drawing calls themselves. Here's a quote:

> Xvnc is a VNC X server. It is based on the standard X server, but it has a "virtual" screen rather than a physical one.
X applications display themselves on it as if it were a regular X display,
but access to them is only possible through a VNC viewer.
Thus, Xvnc is actually two servers in one.
For applications, it's an X server, and for remote VNC users, it's a VNC server.

So, the scheme is as follows: The VNC server launches its own X-server
on a separate display, and this is what gets forwarded via VNC.

On such a virtual display, you can run a simple desktop manager
(I tried LXDE and XFCE), but if you run programs from it,
there's no hardware OpenGL acceleration again!

So, you need to run a separate Xorg for your program on a *different* display.

You also need to install the VirtualGL package, and run the program like this:

`vglrun ./chamferShellGL`

Turbo VNC
---------

AI recommended Turbo VNC as the fastest and most optimized for 3D graphics.
In fact, it's a rather strange program, and everything about it is non-standard,
starting from the installation (an example is in the [Dockerfile](./example.docker)).

Nevertheless, it works quite well. Among the useful features, it uses JPEG compression
with adjustable parameters.

It works best with the Turbo VNC client. I also managed to add VNC support
from the browser, which is called noVNC. It works slower but still acceptable.

Compression methods:

* It has a JPEG compression method, for which the quality can be set.
In my opinion, around 85 is fine. You can also specify whether to do subsampling of color planes,
I use a maximum of 2X.

* It also has a CopyRect compression method. Supposedly, it should help when something moves without changes,
for example, moving a window.

* It also has an Interframe compression mode, in which it should compare with the previous frame
and only send changed parts. If disabled, it will send those parts
that the program redraws on the screen.
In the case of testbedGL, it's usually the entire window, so it's better to enable it.
This method also works a bit strangely, with default huge squares of 256x256.

* There's also a concept called Lossless Refresh. The idea seems good — to request a frame without loss of quality.
Even its client has a button to do a Lossless Refresh. But in practice,
if you press this button once, it will always start sending
very slowly and without loss of quality. There's also an automatic Lossless
Refresh option in its server. The idea is good, for example, every 5 seconds it sends a frame without loss
of quality. Then, stationary areas are drawn well, and moving ones are faster, with JPEG compression.
But in reality, I couldn't get it to work properly.

Tiger VNC
---------

I managed to launch this server both in the mode of capturing an existing X-server and in the mode
of creating a virtual display.

I liked it more in the virtual display mode.

This server doesn't seem to be as optimized as TurboVNC, however, it does exactly what is needed –
it redraws the image without loss of quality during free time.

Launching the container
-----------------------

The container is launched as follows:

```
sudo docker run --rm \
    --net=host \
    --detach \
    --privileged \
    --runtime=nvidia \
    --name example \
```

Of course, instead of net=host, you can expose only the necessary ports.

Inside the container, nvidia-smi should show the presence of a GPU!

After launching, a script like this is executed (the method with a virtual display):

```
export DISPLAY=:0
Xorg :0 -config xorg.conf & # generated config

# TurboVNC
/opt/TurboVNC/bin/vncserver :1 -xstartup ./xstartup.sh -novnc /opt/noVNC

# TigerVNC
# tigervncserver :1 -xstartup ./xstartup.sh -PasswordFile ~/.vnc/passwd -localhost no
```

In the `xstartup.sh` file, you can simply write `startlxde` for the LXDE desktop.

Connecting to VNC
-----------------

* Using a VNC client, for example, TurboVNC.

* From the browser using noVNC:
`http://myhost:5801/vnc.html?host=myhost&port=5901&resize=remote&password=mypassword&autoconnect=true`.
Convenient options are specified: adjusting the remote display size and connecting immediately.
