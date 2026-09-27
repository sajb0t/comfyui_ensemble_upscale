import { app } from "../../scripts/app.js";

function look(noise) {
    return {
        noise: noise,
        blend_mode: "content_aware",
        frequency_split: true,
        texture_smooth: 0,
        reduce_grid: 0,
        photo_filter: "None",
    };
}

const PRESETS = {
    Realistic: {
        models: [
            "4xNomos8kSC.pth",
            "RealESRGAN_x4plus.pth",
            "4x_NMKD-Superscale-SP_178000_G.pth",
        ],
        settings: look(0.03),
    },
    Anime: {
        models: [
            "4x-AnimeSharp.pth",
            "RealESRGAN_x4plus_anime_6B.pth",
            "4x-UltraSharp.pth",
        ],
        settings: look(0),
    },
    Cartoon: {
        models: [
            "4x-UltraSharp.pth",
            "4x_foolhardy_Remacri.pth",
            "4x_NMKD-Siax_200k.pth",
        ],
        settings: look(0),
    },
    Sharp: {
        models: [
            "4x-UltraSharp.pth",
            "4x_foolhardy_Remacri.pth",
            "4xNomos8kSC.pth",
        ],
        settings: look(0.02),
    },
    Smooth: {
        models: [
            "4x_NMKD-Superscale-SP_178000_G.pth",
            "4xNomos8kSC.pth",
            "RealESRGAN_x4plus.pth",
        ],
        settings: look(0.04),
    },
    "2x": {
        models: [
            "RealESRGAN_x2plus.pth",
            "none",
            "none",
        ],
        settings: look(0.03),
    },
};

const CUSTOM = "Custom";
const WATCHED = [
    "model_1_name",
    "model_2_name",
    "model_3_name",
    "noise",
    "blend_mode",
    "frequency_split",
    "texture_smooth",
    "reduce_grid",
    "photo_filter",
];

function widget(node, name) {
    return node.widgets?.find((item) => item.name === name);
}

function setWidget(item, value) {
    if (!item) {
        return;
    }
    const values = item.options?.values;
    if (values && !values.includes(value)) {
        return;
    }
    item.value = value;
}

function applyPreset(node) {
    const preset = widget(node, "preset");
    const picked = PRESETS[preset?.value];
    if (!preset || !picked) {
        return;
    }
    node._ensemblePresetSync = true;
    picked.models.forEach((name, index) => {
        setWidget(widget(node, "model_" + (index + 1) + "_name"), name);
    });
    for (const [name, value] of Object.entries(picked.settings)) {
        setWidget(widget(node, name), value);
    }
    node._ensemblePresetSync = false;
}

function attachPreset(node) {
    const preset = widget(node, "preset");
    if (!preset || !widget(node, "model_1_name") || node._ensemblePresetAttached) {
        return;
    }
    node._ensemblePresetAttached = true;

    const presetCallback = preset.callback;
    preset.callback = function () {
        applyPreset(node);
        return presetCallback?.apply(this, arguments);
    };

    for (const name of WATCHED) {
        const item = widget(node, name);
        if (!item) {
            continue;
        }
        const callback = item.callback;
        item.callback = function () {
            if (!node._ensemblePresetSync && preset.value !== CUSTOM) {
                preset.value = CUSTOM;
            }
            return callback?.apply(this, arguments);
        };
    }
}

app.registerExtension({
    name: "comfyui.ensemble.upscale.presets",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "SmartEnsembleUpscale") {
            return;
        }
        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const result = onNodeCreated?.apply(this, arguments);
            attachPreset(this);
            applyPreset(this);
            return result;
        };
        const onConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function () {
            const result = onConfigure?.apply(this, arguments);
            attachPreset(this);
            applyPreset(this);
            return result;
        };
    },
});
