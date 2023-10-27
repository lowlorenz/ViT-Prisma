# Helper function: javascript visualization
import numpy as np
import json
from IPython.core.display import display, HTML
import string
import random

# Helper function to plot attention patterns, hide this function

import matplotlib.pyplot as plt
import numpy as np

def plot_attn_heads(total_activations, n_heads = 4, n_layers = 1, img_shape=32, idx=0, 
    global_min_max=False, global_normalize=False, fourier_transform_local=False, fourier_transform_global=False, graph_type = "imshow_graph", cmap='viridis'):

    log_transform = False
    save_figure = False

    total_data = np.zeros((n_layers, n_heads, img_shape, img_shape))
    if global_min_max or global_normalize or fourier_transform_global:
        for i in range(n_layers):
            for j in range(n_heads):
                data = total_activations[i][idx,j,:,:]
                if log_transform:
                    data = log10_stable(data)
                if fourier_transform_global:
                    data = np.abs(np.fft.fft2(data))

                total_data[i,j,:,:] = data

        total_min = np.min(total_data)
        total_max = np.max(total_data)
        print(f"Total Min: {total_min}, Total Max: {total_max}")

        if global_normalize:
            total_data = -1 + 2 * (total_data - total_min) / (total_max - total_min)
            # put back in list
            total_activations = []
            for i in range(12):
                matrix = np.expand_dims(total_data[idx,:,:,:], axis=0)
                total_activations.append(matrix)

    fig, axes = plt.subplots(n_layers, n_heads, figsize=(10, 5))
    total_data_dict = {}

    is_1d_axes = len(axes.shape) == 1


    # Iterate over each row and column
    for i in range(n_layers): # Layer

        total_data_dict[f"Layer_{i}"] = {}

        for j in range(n_heads): # Head

            # Get Data
            data = total_activations[i][0,j,:,:]

            # Plot the imshow plot in the corresponding subplot
            if graph_type == "histogram_graph":
                data = data.flatten()
                if log_transform:
                    axes[i, j].set_yscale('log')
                    axes[i, j].set_xscale('log')
    #             axes[i, j].set_title(f"Layer {layer_num}, Head {head_num} Attention")
                n, bins, patches = axes[i, j].hist(data, bins=100)

            elif graph_type == "imshow_graph": # Plot imshow

                if log_transform:
                    data = log10_stable(data)
                    data = np.round(data, 5)
                if fourier_transform_local:
                    data = np.abs(np.fft.fft2(data))

                if global_min_max or fourier_transform_global:
                    vmin = total_min
                    vmax = total_max
                elif global_normalize:
                    vmin = -1
                    vmax = 1
                else:
                    vmax = np.max(data)
                    vmin = np.min(data)
                
                if is_1d_axes:
                    im = axes[j].imshow(np.round(data,5), vmin=vmin, vmax=vmax, cmap=cmap)
                    axes[j].axis('off')
                else:
                    im = axes[i, j].imshow(np.round(data,5), vmin=vmin, vmax=vmax, cmap=cmap)
                    axes[i, j].axis('off')
                    
                total_data_dict[f"Layer_{i}"][f"Head_{j}"] = [[round(num, 5) for num in row] for row in data.tolist()]

                # Remove axis labels and ticks
            if i == 0:
                if is_1d_axes:
                    axes[j].set_title(f"Head {j}", fontsize=12, pad=5)
                else:
                    axes[i, j].set_title(f"Head {j}", fontsize=12, pad=5)
            if j == 0:
                if is_1d_axes:
                    axes[j].text(-0.3, 0.5, f"Layer {i}", fontsize=12, rotation=90, ha='center', va='center', transform=axes[j].transAxes)
                else:
                    axes[i, j].text(-0.3, 0.5, f"Layer {i}", fontsize=12, rotation=90, ha='center', va='center', transform=axes[i, j].transAxes)

    # Add colorbar
    if graph_type == "imshow_graph" and global_min_max:
        cbar_ax = fig.add_axes([0.15, 0.95, 0.7, 0.01])
        # Create a colorbar in the new axes
        cbar = fig.colorbar(im, cax=cbar_ax, orientation='horizontal', shrink=.4)
        if global_min_max:
            cbar.set_label('Color Scale Across All Attention Heads', fontsize=16)
        else:
            cbar.set_label('Color Scale for Each Individual Attention Head', fontsize=16)

    # Adjust the spacing between subplots
    if graph_type == "histogram_graph":
        plt.subplots_adjust(wspace=0.6, hspace=0.6)
    elif graph_type == "imshow_graph":
        plt.subplots_adjust(wspace=0.1, hspace=0.2)

    # if global_min_max:
    #     plt.suptitle(f'Attention Scores for ViT on CIFAR-10 for Image Idx {idx}', fontsize=20, y=.98)
    # else:
    #     plt.suptitle(f'Attention Scores for ViT on CIFAR-10 for Image Idx {idx} (Colorbar per Image, not Global)', fontsize=20, y=.98)

    # plt.tight_layout()
    # save_path = './figures/attn_pattern_imshow_log_globalcolor.png'
    # save_path = './figures/attn_pattern_imshow_log.png'
    # if save_figure:
    #     save_path = './figures/attn_scores_truck_imshow_log_globalcolor.png'
    #     plt.savefig(save_path, bbox_inches='tight')
    #     print(f"Saved at {save_path}")

    plt.show()

def generate_random_string(length=10):
    '''
    Helper function to generate canvas IDs for javascript figures.
    '''
    letters = string.ascii_lowercase
    return ''.join(random.choice(letters) for i in range(length))

def plot_javascript(attn_head, image):
    '''
    Attention head is full 50x50 matrix
    Image is 3x224x224
    '''

    num_patches = len(attn_head) - 1 # -1 without CLS token
    image_size = len(image[-1])
    patch_size = int(image_size // np.sqrt(num_patches))

    print("num patches", num_patches)
    print("image_size", image_size)
    print("patch_size", patch_size)

    canvas_img_id = generate_random_string()
    canvas_attn_id = generate_random_string()

    cifar_image = image.numpy()
    cifar_image = (cifar_image - cifar_image.min()) / (cifar_image.max() - cifar_image.min()) * 255
    cifar_image = cifar_image.astype('uint8')

    # Reshape to (224, 224, 3) to make it channel-last
    cifar_image = np.transpose(cifar_image, (1, 2, 0))

    # Create patches of size (32, 32, 3)
    patches = [cifar_image[i:i+patch_size, j:j+patch_size, :] for i in range(0, image_size, patch_size) for j in range(0, image_size, patch_size)]

    # Flatten each patch and create a list of these flattened patches
    flattened_patches = [patch.flatten().tolist() for patch in patches]

    # Convert to JSON
    patches_json = json.dumps(flattened_patches)

    # Assuming attn_head is a 49x49 numpy array
    # (Make sure this is properly set up before running the code)
    # Assuming attn_head is a 2D numpy array of shape (49, 49)
    min_val = np.min(attn_head)
    max_val = np.max(attn_head)

    # Normalize the attention values between 0 and 1
    normalized_attn_head = (attn_head - min_val) / (max_val - min_val)

    # Convert to JSON (if needed)
    attn_head_json = json.dumps(normalized_attn_head[1:, 1:].tolist())


    # HTML and JavaScript code to render the patches on a canvas

    ATTN_SCALING = 8

    html_code = f"""
    <div style="display: flex;">
        <canvas id="{canvas_attn_id}" width="{num_patches*ATTN_SCALING}" height="{num_patches*ATTN_SCALING}" style="width:{num_patches*ATTN_SCALING+20}px; height:{num_patches*ATTN_SCALING+20}px;"></canvas>
        <canvas id="{canvas_img_id}" width="{image_size}" height="{image_size}" style="width:{image_size}px; height:{image_size}px;"></canvas>
    </div>
    <script>

        function patchToImageData(patch, width, height) {{
            var imgData = new ImageData(width, height);
            var data = imgData.data;
            for (let p = 0, q = 0; p < patch.length; p += 3, q += 4) {{
                data[q] = patch[p];
                data[q + 1] = patch[p + 1];
                data[q + 2] = patch[p + 2];
                data[q + 3] = 255;
            }}
            return imgData;
        }}

        // Function to generate a pastel color based on an input
        function generatePastelColor(baseRed = 255, baseGreen = 0, baseBlue = 0) {{
            var red = Math.floor((255 + baseRed) / 2);
            var green = Math.floor((255 + baseGreen) / 2);
            var blue = Math.floor((255 + baseBlue) / 2);
            return `rgba(${{red}}, ${{green}}, ${{blue}}, 1.0)`; // 0.7 is the alpha value for translucency
        }}


        function getColor(intensity) {{
        const viridisColorMap = [
            {{pos: 0, rgb: [68, 1, 84]}} ,
            {{pos: 0.1, rgb: [72, 34, 115]}},
            {{pos: 0.2, rgb: [64, 67, 135]}},
            {{pos: 0.3, rgb: [52, 94, 141]}},
            {{pos: 0.4, rgb: [41, 120, 142]}},
            {{pos: 0.5, rgb: [32, 144, 140]}},
            {{pos: 0.6, rgb: [34, 167, 132]}},
            {{pos: 0.7, rgb: [68, 190, 112]}},
            {{pos: 0.8, rgb: [121, 209, 81]}},
            {{pos: 0.9, rgb: [189, 222, 38]}},
            {{pos: 1.0, rgb: [253, 231, 37]}}
        ];

        for (let i = 0; i < viridisColorMap.length - 1; i++) {{
            const start = viridisColorMap[i];
            const end = viridisColorMap[i + 1];
            if (intensity >= start.pos && intensity < end.pos) {{
                const ratio = (intensity - start.pos) / (end.pos - start.pos);
                const r = Math.floor(start.rgb[0] + ratio * (end.rgb[0] - start.rgb[0]));
                const g = Math.floor(start.rgb[1] + ratio * (end.rgb[1] - start.rgb[1]));
                const b = Math.floor(start.rgb[2] + ratio * (end.rgb[2] - start.rgb[2]));
                return `rgba(${{r}}, ${{g}}, ${{b}}, 1.0)`;
            }}
        }}
        return `rgba(253, 231, 37, 1.0)`;
    }}

        var colorTokenA = 'rgba(0, 128, 128, 0.8)'; //teal
        var colorTokenB = 'rgba(255, 105, 180, 0.7)'; //pink

        var lastHighlightedCol = null;
        var lastHighlightedColSecond = null;

        var matrixColorsImg = Array({num_patches}).fill().map(() => Array({num_patches}).fill('')); // cifar image
        var matrixColorsAttn = Array({num_patches**2}).fill().map(() => Array({num_patches**2}).fill('')); // attention head

        // PLOT CIFAR on canvasImg
        var patches = JSON.parse('{patches_json}');
        var canvasImg = document.getElementById('{canvas_img_id}');
        var ctxImg = {canvas_img_id}.getContext('2d');
        var idx = 0;
        for (let i = 0; i < {image_size}; i+= {patch_size}) {{
            for (let j = 0; j < {image_size}; j += {patch_size}) {{
                var imgData = ctxImg.createImageData({patch_size}, {patch_size});
                var data = imgData.data;
                var patch = patches[idx];

                for (let p = 0, q = 0; p < patch.length; p += 3, q += 4) {{
                    data[q] = patch[p];
                    data[q + 1] = patch[p + 1];
                    data[q + 2] = patch[p + 2];
                    data[q + 3] = 255;
                }}

                const row = Math.floor(i / {patch_size});
                const col = Math.floor(j / {patch_size});

                // Storing the representative color for this patch.
                // You can use the first pixel as a representative color, or calculate the average color of the patch.
                matrixColorsImg[row][col] = patch

                ctxImg.putImageData(imgData, j, i);
                ctxImg.strokeStyle = 'white';
                ctxImg.strokeRect(j, i, {patch_size}, {patch_size});

                idx++;
            }}
        }}

        // Plot attention head on canvasAttn
        var attn_head = JSON.parse('{attn_head_json}');
        var canvasAttn = document.getElementById('{canvas_attn_id}');
        var ctxAttn = {canvas_attn_id}.getContext('2d');



        for (let i = 0; i < {num_patches}; i++) {{
            for (let j = 0; j < {num_patches}; j++) {{
                // var intensity = attn_head[i][j];
                // var red = Math.floor(intensity * 255);
                // var color = generatePastelColor(red, 0, 0);
                var color = getColor(attn_head[i][j]);
                ctxAttn.fillStyle = color;

                // ctxAttn.fillStyle = `rgba(${{255 * attn_head[i][j]}}, 0, 0, 1)`;
                ctxAttn.fillRect(j * 8, i * 8, 8, 8);
                // matrixColorsAttn[i][j] = `rgba(${{255 * attn_head[i][j]}}, 0, 0, 1)`;
                matrixColorsAttn[i][j] = color;
            }}
        }}

        // Add listeners for highlighted pixels

    canvasAttn.addEventListener('mousemove', function(event) {{
            const x = Math.floor(event.offsetY / 8);
            const rowImg = Math.floor(x / {num_patches});
            const colImg = x % {num_patches};

            const y = Math.floor(event.offsetX / 8);
            const rowImgSecond = Math.floor(y / {num_patches});
            const colImgSecond = y % {num_patches};

        if (lastHighlightedCol !== null) {{
            const prevrowImg = Math.floor(lastHighlightedCol / {num_patches});
            const prevcolImg = lastHighlightedCol % {num_patches};
            var originalPatch = matrixColorsImg[prevrowImg][prevcolImg];
            var imgData = patchToImageData(originalPatch, {patch_size}, {patch_size});

            ctxImg.putImageData(imgData, prevcolImg * {patch_size}, prevrowImg * {patch_size});
            ctxImg.strokeStyle = 'white';
            ctxImg.strokeRect(prevcolImg*{patch_size}, prevrowImg*{patch_size}, {patch_size}, {patch_size});

            // Fill in attn matrix
            ctxAttn.fillStyle = matrixColorsAttn[lastHighlightedCol][lastHighlightedColSecond];
            ctxAttn.fillRect(lastHighlightedColSecond * 8, lastHighlightedCol * 8, 8, 8);

        }}

        if (lastHighlightedColSecond !== null) {{
            const prevrowImg = Math.floor(lastHighlightedColSecond / {num_patches});
            const prevcolImg = lastHighlightedColSecond % {num_patches};
            var originalPatch = matrixColorsImg[prevrowImg][prevcolImg];
            var imgData = patchToImageData(originalPatch, {patch_size}, {patch_size});

            ctxImg.putImageData(imgData, prevcolImg * {patch_size}, prevrowImg * {patch_size});
            ctxImg.strokeStyle = 'white';
            ctxImg.strokeRect(prevcolImg*{patch_size}, prevrowImg*{patch_size}, {patch_size}, {patch_size});

        }}

        lastHighlightedCol = x;
        lastHighlightedColSecond = y;  //

        ctxImg.fillStyle = colorTokenA;
        ctxImg.fillRect(colImg * {patch_size}, rowImg * {patch_size}, {patch_size}, {patch_size});

        ctxImg.fillStyle = colorTokenB;
        ctxImg.fillRect(colImgSecond * {patch_size}, rowImgSecond * {patch_size}, {patch_size}, {patch_size});  // Second highlighted pixel

        ctxAttn.fillStyle = 'white';
        ctxAttn.fillRect(y * 8, x * 8, 8, 8);

        }}, {{ passive: true }});

        canvasAttn.addEventListener('mouseout', function() {{
            if (lastHighlightedCol !== null) {{

                const prevrowImg = Math.floor(lastHighlightedCol / {num_patches});
                const prevcolImg = lastHighlightedCol % {num_patches};

                if (matrixColorsImg[prevrowImg] && matrixColorsImg[prevrowImg][prevcolImg]) {{

                    // Fill in rectangle for img
                    var originalPatch = matrixColorsImg[prevrowImg][prevcolImg];
                    var imgData = patchToImageData(originalPatch, {patch_size}, {patch_size});
                    ctxImg.putImageData(imgData, prevcolImg * {patch_size}, prevrowImg * {patch_size});
                    ctxImg.strokeStyle = 'white';
                    ctxImg.strokeRect(prevcolImg * {patch_size}, prevrowImg * {patch_size}, {patch_size}, {patch_size});

                    // Fill in attn matrix
                    ctxAttn.fillStyle = matrixColorsAttn[lastHighlightedCol][lastHighlightedColSecond];
                    ctxAttn.fillRect(lastHighlightedColSecond * 8, lastHighlightedCol * 8, 8, 8);
                }}
            }}

            if (lastHighlightedColSecond !== null) {{
                const prevrowImg = Math.floor(lastHighlightedColSecond / {num_patches});
                const prevcolImg = lastHighlightedColSecond % {num_patches};

                if (matrixColorsImg[prevrowImg] && matrixColorsImg[prevrowImg][prevcolImg]) {{
                    // Fill in rectangle for img
                    var originalPatch = matrixColorsImg[prevrowImg][prevcolImg];
                    var imgData = patchToImageData(originalPatch, {patch_size}, {patch_size});
                    ctxImg.putImageData(imgData, prevcolImg * {patch_size}, prevrowImg * {patch_size});
                    ctxImg.strokeStyle = 'white';
                    ctxImg.strokeRect(prevcolImg * {patch_size}, prevrowImg * {patch_size}, {patch_size}, {patch_size});

                    // Fill in attn matrix
                    ctxAttn.fillStyle = matrixColorsAttn[lastHighlightedCol][lastHighlightedColSecond];
                    ctxAttn.fillRect(lastHighlightedColSecond * 8, lastHighlightedCol * 8, 8, 8);
                }}

            }}

            lastHighlightedCol = null;
            lastHighlightedColSecond = null;  // Reset this too

        }}, {{ passive: true }});



    </script>
    """

    return html_code
