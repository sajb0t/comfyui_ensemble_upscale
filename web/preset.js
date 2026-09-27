import { app } from "../../scripts/app.js";

const PRESETS = {
    Realistic: [
        "4xNomos8kSC.pth",
        "RealESRGAN_x4plus.pth",
        "4x_NMKD-Superscale-SP_178000_G.pth",
    ],
    Anime: [
        "4x-AnimeSharp.pth",
        "RealESRGAN_x4plus_anime_6B.pth",
        "4x-UltraSharp.pth",
    ],
    Sharp: [
        "4x-UltraSharp.pth",
        "4x_foolhardy_Remacri.pth",
        "4xNomos8kSC.pth",
    ],
    Smooth: [
        "4x_NMKD-Superscale-SP_178000_G.pth",
        "4xNomos8kSC.pth",
        "RealESRGAN_x4plus.pth",
    ],
    "2x": [
        "RealESRGAN_x2plus.pth",
        "none",
        "none",
    ],
};

const CUSTOM = "Custom";

function widget(node, name) {
    return node.widgets?.find((item) => item.name === name);
}

function applyPreset(node) {
    const preset = widget(node, "preset");
    const models = ["model_1_name", "model_2_name", "model_3_name"].map((name) => widget(node, name));
    const picked = PRESETS[preset?.value];
    if (!preset || !picked || models.some((item) => !item)) {
        return;
    }
    node._ensemblePresetSync = true;
    models.forEach((item, index) => {
        const next = picked[index];
        const values = item.options?.values;
        if (!values || values.includes(next)) {
            item.value = next;
        }
    });
    node._ensemblePresetSync = false;
}

function attachPreset(node) {
    const preset = widget(node, "preset");
    const models = ["model_1_name", "model_2_name", "model_3_name"].map((name) => widget(node, name));
    if (!preset || models.some((item) => !item) || node._ensemblePresetAttached) {
        return;
    }
    node._ensemblePresetAttached = true;

    const presetCallback = preset.callback;
    preset.callback = function () {
        applyPreset(node);
        return presetCallback?.apply(this, arguments);
    };

    for (const item of models) {
        const modelCallback = item.callback;
        item.callback = function () {
            if (!node._ensemblePresetSync && preset.value !== CUSTOM) {
                preset.value = CUSTOM;
            }
            return modelCallback?.apply(this, arguments);
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
