"""``sidecar`` package — FaceFusion HTTP sidecar for LiveEmote.

See :mod:`sidecar.facefusion_runner` for the lazy-loaded FaceFusion seam
and :mod:`sidecar.app` for the FastAPI surface. The main LiveEmote process
(:mod:`packages.hermes_avatar.renderer.facefusion_sidecar_daemon`) talks
to this service over HTTP instead of importing the vendored FaceFusion
tree in-process.
"""
