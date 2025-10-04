import { app } from "../../scripts/app.js";
import { load } from "./lib/npyjs.js";
import { reshape } from "./lib/reshape.js";

let tooltipDiv = document.createElement("div");
tooltipDiv.id = "daam-node-tooltip";

// Basic styling for the tooltip
Object.assign(tooltipDiv.style, {
    position: "absolute",

    background: "#1e1e1e",
    color: "#f0f0f0",
    border: "1px solid #444",

    // Layout and Typography
    padding: "10px 14px",
    borderRadius: "6px",
    fontSize: "13px",
    fontFamily: "Inter, sans-serif",
    lineHeight: "1.4",

    // Positioning and Rendering
    zIndex: "10000",
    pointerEvents: "none",
    display: "none",
    whiteSpace: "nowrap",
    boxShadow: "0 6px 12px rgba(0, 0, 0, 0.4)",
    maxWidth: "300px",
    textAlign: "left",
    minWidth: "120px",
});

document.body.appendChild(tooltipDiv);

app.registerExtension({
    name: "comfyui_daam.daam_preview_image",

    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeType.comfyClass === "DAAMPreviewImage") {
            // Store the original draw method only once when node is created
            const onDrawBackground = nodeType.prototype.onDrawBackground;

            // onDrawBackground is created to keep track of where the displaying image is drawn relative to the node and its size
            nodeType.prototype.onDrawBackground = function () {
                const canvas = arguments[0]; // CanvasRenderingContext2D
                const self = this;

                // Initialize the node map if it doesn't exist
                if (!canvas._daamNodeMap) {
                    canvas._daamNodeMap = new Map();
                }

                // Register this node if it has images
                if (self.imgs && self.imgs.length > 0) {
                    canvas._daamNodeMap.set(self.id, self);
                } else {
                    canvas._daamNodeMap.delete(self.id);
                }

                // Set up the drawImage hook only once globally
                if (!canvas._originalDrawImage) {
                    canvas._originalDrawImage = canvas.drawImage;
                    canvas.drawImage = function () {
                        const drawnImage = arguments[0];

                        // Check all registered nodes to see if this image belongs to them
                        for (const [
                            nodeId,
                            node,
                        ] of canvas._daamNodeMap.entries()) {
                            if (
                                node.imgs &&
                                node.imgs.length > 0 &&
                                drawnImage === node.imgs[node.imageIndex || 0]
                            ) {
                                node._imageRect = {
                                    x: arguments[1],
                                    y: arguments[2],
                                    width: arguments[3],
                                    height: arguments[4],
                                };
                            }
                        }
                        return canvas._originalDrawImage.apply(this, arguments);
                    };
                }

                onDrawBackground?.apply(this, arguments);
            };

            // Once the node is executed, fetch and parse the heatmap data and tokenized words
            const onExecuted = nodeType.prototype.onExecuted;
            nodeType.prototype.onExecuted = async function (message) {
                onExecuted?.apply(this, arguments);

                const self = this;
                const heatmap_files = message?.heatmap || [];

                const fetchPromises = heatmap_files.map(
                    async (heatmap_file) => {
                        const { filename, subfolder, type } =
                            heatmap_file || {};

                        if (!filename) return null;

                        const heatmap_src = `/view?filename=${encodeURIComponent(
                            filename
                        )}&type=${type}&subfolder=${encodeURIComponent(
                            subfolder
                        )}`;

                        try {
                            const response = await fetch(heatmap_src);
                            if (!response.ok) {
                                console.error(
                                    `Failed to fetch heatmap: ${filename}. Status: ${response.status}`
                                );
                                return null;
                            }

                            let arrayBuffer = await response.arrayBuffer();
                            const { data, shape, fortranOrder } = await load(
                                arrayBuffer
                            );
                            const heatmap = reshape(data, shape, fortranOrder);

                            arrayBuffer = null;

                            return heatmap;
                        } catch (error) {
                            console.error(
                                `Error processing heatmap ${filename}:`,
                                error
                            );
                            return null;
                        }
                    }
                );

                const heatmaps_data = await Promise.all(fetchPromises);

                self.heatmaps = heatmaps_data;
                self.attention_words = message?.attention_words || [];
            };
        }
    },

    async nodeCreated(node) {
        if (node?.comfyClass === "DAAMPreviewImage") {
            const original_onMouseEnter = node.onMouseEnter;
            const original_onMouseMove = node.onMouseMove;
            const original_onMouseLeave = node.onMouseLeave;

            node.onMouseEnter = function (event) {
                if (this.imgs && this.imgs.length > 0) {
                    original_onMouseEnter?.apply(this, arguments);

                    tooltipDiv.style.display = "block";
                    updateTooltip(this, event);
                }
            };

            node.onMouseMove = function (event) {
                if (this.imgs && this.imgs.length > 0) {
                    original_onMouseMove?.apply(this, arguments);

                    if (tooltipDiv.style.display === "block") {
                        updateTooltip(this, event);
                    }
                }
            };

            node.onMouseLeave = function (event) {
                if (this.imgs && this.imgs.length > 0) {
                    original_onMouseLeave?.apply(this, arguments);

                    tooltipDiv.style.display = "none";
                }
            };

            /**
             * Updates the tooltip's content and position based on the mouse event.
             * @param {LGraphNode} currentNode The node being hovered.
             * @param {MouseEvent} event The mouse event object.
             */
            function updateTooltip(currentNode, event) {
                const canvas = app.canvas;

                // Calculate node position relative to the canvas
                const imgOffset = currentNode._imageRect; // Offset from the node's top-left corner

                const imgCanvasPosition = {
                    x: currentNode.pos[0] + imgOffset.x,
                    y: currentNode.pos[1] + imgOffset.y,
                };

                const diff = {
                    x: canvas.graph_mouse[0] - imgCanvasPosition.x,
                    y: canvas.graph_mouse[1] - imgCanvasPosition.y,
                };

                const percentX =
                    imgOffset.width > 0 ? diff.x / imgOffset.width : 0;
                const percentY =
                    imgOffset.height > 0 ? diff.y / imgOffset.height : 0;

                const clampedX = Math.max(0, Math.min(1, percentX));
                const clampedY = Math.max(0, Math.min(1, percentY));

                const rescaledX = Math.min(
                    Math.floor(clampedX * currentNode.imgs[0].width),
                    currentNode.imgs[0].width - 1
                );
                const rescaledY = Math.min(
                    Math.floor(clampedY * currentNode.imgs[0].height),
                    currentNode.imgs[0].height - 1
                );

                if (
                    currentNode.imgs.length === 1 ||
                    currentNode.imageIndex != null
                ) {
                    let contentArray = [];

                    if (
                        currentNode.heatmaps &&
                        currentNode.heatmaps.length > 0
                    ) {
                        const top_token_indices =
                            currentNode.heatmaps[currentNode.imageIndex || 0][
                                rescaledY
                            ][rescaledX];
                        const processed_attn_words = top_token_indices.map(
                            (idx) => {
                                let word = currentNode.attention_words[idx];
                                // Highlight full words (those ending with </w>)
                                const is_full_word = word.endsWith("</w>");
                                if (is_full_word) {
                                    word = word.slice(0, -4);
                                    return `<b>${word}</b>`;
                                }
                                return word;
                            }
                        );

                        contentArray.push(...processed_attn_words);
                    }

                    const coordsHtml = `<div style="text-align: right; color: #aaa; font-size: 10px;">(${rescaledX}, ${rescaledY})</div>`;
                    contentArray.push(coordsHtml);

                    tooltipDiv.innerHTML = contentArray.join("<br>");

                    // Position the tooltip near the cursor
                    tooltipDiv.style.left = `${event.clientX + 15}px`;
                    tooltipDiv.style.top = `${event.clientY + 15}px`;
                }
            }
        }
    },
});
