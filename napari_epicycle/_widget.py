from napari.viewer import Viewer

from .ui_shell import EpiCycleWorkflowShell


class EpiCycleMainWidget(EpiCycleWorkflowShell):
    """Compatibility entrypoint for napari plugin manifest."""

    def __init__(self, napari_viewer: Viewer):
        super().__init__(napari_viewer=napari_viewer)


# Plugin hook

def napari_experimental_provide_dock_widget():
    return EpiCycleMainWidget
