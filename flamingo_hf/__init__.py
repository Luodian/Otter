from typing import TYPE_CHECKING

from transformers.utils import (
    OptionalDependencyNotAvailable,
    _LazyModule,
    is_torch_available,
)


_import_structure = {
    "configuration_flamingo": [
        "FlamingoConfig",
    ],
    # "processing_flamingo": ["FlamingoProcessor"],
}

try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["modeling_flamingo"] = [
        # "BLIP_2_PRETRAINED_MODEL_ARCHIVE_LIST",
        "FlamingoModel",
        "FlamingoPreTrainedModel",
        "FlamingoForConditionalGeneration",
    ]

if TYPE_CHECKING:
    from .configuration_flamingo import (
        # BLIP_2_PRETRAINED_CONFIG_ARCHIVE_MAP,
        FlamingoConfig,
    )

    # from .processing_flamingo import FlamingoProcessor

    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .modeling_flamingo import (
            # BLIP_2_PRETRAINED_MODEL_ARCHIVE_LIST,
            FlamingoForConditionalGeneration,
            FlamingoModel,
            FlamingoPreTrainedModel,
        )

else:
    import sys

    sys.modules[__name__] = _LazyModule(
        __name__, globals()["__file__"], _import_structure, module_spec=__spec__
    )
