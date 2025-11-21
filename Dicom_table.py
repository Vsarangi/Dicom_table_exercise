def get_metadata_for_table(ds):
    instance_uid = ds.SOPInstanceUID
    instance_number = ds.InstanceNumber
    number_of_frames = ds.NumberOfFrames
    pixel_spacing = ds.SharedFunctionalGroupsSequence[0].PixelMeasuresSequence[0].PixelSpacing
    stain = ds.SpecimenDescriptionSequence[0].SpecimenPreparationSequence[0].SpecimenPreparationStepContentItemSequence[
        2].ConceptNameCodeSequence[0].CodeValue
    return instance_uid, instance_number, number_of_frames, pixel_spacing, stain


def format_frame(ds, frame_list):
    dtype = np.uint16 if ds.BitsAllocated == 16 else np.uint8
    frame = np.frombuffer(frame_list[0], dtype=dtype)

    if ds.SamplesPerPixel > 1:
        frame = frame.reshape(ds.Rows, ds.Columns, ds.SamplesPerPixel)
    else:
        frame = frame.reshape(ds.Rows, ds.Columns)
    return frame


def ostu_threshold_method(arr):
    arr_f = arr.astype(np.float32) / 255.0
    brightness = arr_f.mean(-1)
    saturation = arr_f.max(-1) - arr_f.min(-1)
    gate = (brightness < 0.95) & (saturation > 0.03)
    thr = threshold_otsu(rgb2gray(arr_f))
    mask_binary = rgb2gray(arr_f) < thr
    mask_binary &= gate
    return mask_binary


def get_frame_with_tissue(low_res_instance):
    low_res_frames = low_res_instance.NumberOfFrames
    best_mask_score = 0
    chosen_frame = 2
    best_average_pixel = 0
    for frame_index in range(1, low_res_frames + 1):

        frame_list = client.retrieve_instance_frames(study_instance_uid=study_id,
                                                     series_instance_uid=series_id,
                                                     sop_instance_uid=low_res_instance_ID,
                                                     frame_numbers=[frame_index]
                                                     )
        frame = format_frame(low_res_instance, frame_list)

        mask_binary = ostu_threshold_method(frame)
        percent_mask = np.mean(mask_binary)

        if percent_mask > best_mask_score:
            best_mask_score = percent_mask
            chosen_frame = frame_index
            best_average_pixel = np.mean(frame)
    return best_mask_score, chosen_frame, best_average_pixel


def tiled_full_grid_facts(ds):
    TPM_rows = int(ds.TotalPixelMatrixRows)
    TPM_cols = int(ds.TotalPixelMatrixColumns)
    tile_rows = int(ds.Rows)  # tile height  (pixels)
    tile_cols = int(ds.Columns)  # tile width   (pixels)
    tiles_y = math.ceil(TPM_rows / tile_rows)
    tiles_x = math.ceil(TPM_cols / tile_cols)
    return TPM_rows, TPM_cols, tile_rows, tile_cols, tiles_y, tiles_x


def frame_coord_to_totalpixel_coord(instance, frame_index, x_in_tile, y_in_tile):
    # returns x,y pixel coordinate

    TPM_rows, TPM_cols, tile_rows, tile_cols, tiles_y, tiles_x = tiled_full_grid_facts(instance)

    tile_row = (frame_index - 1) // tiles_x
    tile_col = (frame_index - 1) % tiles_x

    x_total = tile_row * tile_rows + x_in_tile
    y_total = tile_col * tile_cols + y_in_tile
    return x_total, y_total


def map_low_pixel_to_high(x_low, y_low, low_res_instance, high_res_instance):
    dx_low, dy_low = low_res_instance.SharedFunctionalGroupsSequence[0].PixelMeasuresSequence[0].PixelSpacing
    dx_high, dy_high = high_res_instance.SharedFunctionalGroupsSequence[0].PixelMeasuresSequence[0].PixelSpacing

    origin_low_x_mm = low_res_instance.TotalPixelMatrixOriginSequence[0].XOffsetInSlideCoordinateSystem
    origin_low_y_mm = low_res_instance.TotalPixelMatrixOriginSequence[0].YOffsetInSlideCoordinateSystem

    origin_high_x_mm = high_res_instance.TotalPixelMatrixOriginSequence[0].XOffsetInSlideCoordinateSystem
    origin_high_y_mm = high_res_instance.TotalPixelMatrixOriginSequence[0].YOffsetInSlideCoordinateSystem

    # Physical coordinate of low-res pixel
    X_mm = origin_low_x_mm + x_low * dx_low
    Y_mm = origin_low_y_mm + y_low * dy_low

    # Convert to high-res coordinates
    x_high = int(round((X_mm - origin_high_x_mm) / dx_high))
    y_high = int(round((Y_mm - origin_high_y_mm) / dy_high))

    # Compute coverage (how many high-res pixels correspond)
    scale_x = dx_low / dx_high
    scale_y = dy_low / dy_high

    x_range = range(x_high, x_high + math.ceil(scale_x))
    y_range = range(y_high, y_high + math.ceil(scale_y))

    return x_high, y_high, list(x_range), list(y_range)


def get_frames_index_from_x_y(instance, x0, y0, x1, y1):
    # returns frame index

    TPM_rows, TPM_cols, tile_rows, tile_cols, tiles_y, tiles_x = tiled_full_grid_facts(instance)

    tr0 = x0 // tile_rows
    tr1 = x1 // tile_rows
    tc0 = y0 // tile_cols
    tc1 = y1 // tile_cols

    hits = []
    frame_coords = []
    for tr in range(tr0, tr1 + 1):
        for tc in range(tc0, tc1 + 1):
            frame = (tr * tiles_x + tc) + 1
            hits.append(frame)

            x_tile_origin = tr * tile_rows
            y_tile_origin = tc * tile_cols

            x_in_tile = x0 - x_tile_origin
            y_in_tile = y0 - y_tile_origin

            frame_coords.append([x_in_tile, y_in_tile])

    return hits, frame_coords


import pydicom
from io import BytesIO
import io
from pydicom.filereader import dcmread
import google.auth
from google.auth.transport import requests
from dicomweb_client.api import DICOMwebClient
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from skimage import data
from skimage.filters import threshold_otsu
from skimage.color import rgb2gray
from skimage import img_as_float
import math
import matplotlib.patches as patches

credentials, _ = google.auth.default()

session = requests.AuthorizedSession(credentials)

project_id = 'ml-mps-adl-dpp-ndsa-p-4863'
location = 'us'
dataset_id = 'ml-phi-pathology-data-us-p'
dicom_store_id = 'ml-phi-pathology-data-us-p-dicom-ndsa'
headers = {"Accept": "application/dicom+json"}

client = DICOMwebClient(
    url=(
        f"https://healthcare.googleapis.com/v1/"
        f"projects/{project_id}/locations/{location}/datasets/{dataset_id}/"
        f"dicomStores/{dicom_store_id}/dicomWeb"
    ),
    session=session,
    headers=headers
)

study_uid = "1.2.840.113713.1.1119.119276.19276865414512"

studies = client.search_for_studies(search_filters={"0020000D": study_uid})

output_file = open("Dicom_exercise_table.tsv", 'w')
output_file.write("\t".join(
    ["Study", "Series", "Instance", "Instance_Number", "Pixel_Spacing", "Number_of_Frames", "Stain",
     "Average_Pixel_Value"]) + "\n")

# Iterating through all studies:
for study in studies:
    study = pydicom.dataset.Dataset.from_json(study)
    study_id = study.StudyInstanceUID
    series_list = client.search_for_series(study_id)

    # Iterating through all series in the Study
    for series_count, series in enumerate(series_list):
        print("images/Series_" + str(series_count))
        series = pydicom.dataset.Dataset.from_json(series)
        series_id = series.SeriesInstanceUID
        instances = client.search_for_instances(
            study_instance_uid=study_id,
            series_instance_uid=series_id
        )
        volume_dict = {}

        # Iterating through all instances within the series to find the high resolution instance and one low resolution instance (2 level of downsampling).
        for instance in instances:
            instance = pydicom.dataset.Dataset.from_json(instance)
            instance_id = instance.SOPInstanceUID
            metadata_instance = client.retrieve_instance_metadata(study_instance_uid=study_id,
                                                                  series_instance_uid=series_id,
                                                                  sop_instance_uid=instance_id)

            metadata_instance = pydicom.dataset.Dataset.from_json(metadata_instance)
            image_type = metadata_instance.ImageType
            number_of_frames = metadata_instance.NumberOfFrames

            if image_type[2] == "VOLUME":
                volume_dict[number_of_frames] = instance_id

        ## Using the lowest resolution to get frames with tissues:
        print(sorted(volume_dict))
        low_res_instance_ID = volume_dict[
            sorted(volume_dict)[0]]  ### this -5 is not ideal and I need to extract it from the metadata somehow.
        low_res_instance = pydicom.dataset.Dataset.from_json(
            client.retrieve_instance_metadata(study_instance_uid=study_id,
                                              series_instance_uid=series_id,
                                              sop_instance_uid=low_res_instance_ID
                                              ))
        instance_uid, instance_number, number_of_frames, pixel_spacing, stain = get_metadata_for_table(low_res_instance)

        ## Getting the frame with most tissue pixel and retreiving the frame instance
        best_mask_score, chosen_frame, best_average_pixel = get_frame_with_tissue(
            low_res_instance)  # gets the frame with most tissue pixels
        line = "\t".join(
            [study_id, series_id, instance_uid, str(instance_number), str(pixel_spacing), str(number_of_frames), stain,
             str(best_average_pixel)])
        output_file.write(line + "\n")
        frame_list = client.retrieve_instance_frames(study_instance_uid=study_id,
                                                     series_instance_uid=series_id,
                                                     sop_instance_uid=low_res_instance_ID,
                                                     frame_numbers=[chosen_frame]
                                                     )

        # Converting frame list to ndarray
        frame = format_frame(low_res_instance, frame_list)

        # identifying a tissue pixel in the frame
        mask_binary = ostu_threshold_method(frame)
        pixel_with_tissue = np.where(mask_binary)
        location = 100
        x_in_tile = pixel_with_tissue[0][location]
        y_in_tile = pixel_with_tissue[1][location]

        # Plotting set up (use constrained_layout to reduce whitespace)
        n_rows = 1
        n_cols = len(sorted(volume_dict)[1:]) + 1
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 10), constrained_layout=True)
        axes_flat = axes.flatten()
        img = axes_flat[0].imshow(frame)
        rect = patches.Rectangle((y_in_tile - 2, x_in_tile - 2), 3, 3, linewidth=0.5, edgecolor='r', facecolor='red',
                                 alpha=0.7)
        axes_flat[0].add_patch(rect)
        axes_flat[0].set_title("Pixel_Spacing" + str(pixel_spacing))

        # get totol pixel coordinate for low res
        x_low, y_low = frame_coord_to_totalpixel_coord(low_res_instance, chosen_frame, x_in_tile, y_in_tile)
        # print(x_low, y_low)

        # using all other high resolution volume to map the low res pixel from above

        for n, high_res_instance_ID_index in enumerate(sorted(volume_dict)[1:]):

            high_res_instance_ID = volume_dict[high_res_instance_ID_index]
            high_res_instance = pydicom.dataset.Dataset.from_json(
                client.retrieve_instance_metadata(study_instance_uid=study_id,
                                                  series_instance_uid=series_id,
                                                  sop_instance_uid=high_res_instance_ID))

            # Get total pixel coordinate for high res
            x_high, y_high, x_range, y_range = map_low_pixel_to_high(x_low, y_low, low_res_instance, high_res_instance)
            # print(x_high, y_high, x_range, y_range)

            ## get frame hits from pixel coordinates
            x0 = x_range[0]
            x1 = x_range[-1]
            y0 = y_range[0]
            y1 = y_range[-1]
            # print(x0,y0,x1,y1)

            frame_hits, tile_xy_coord = get_frames_index_from_x_y(high_res_instance, x0, y0, x1, y1)
            hit_frame = frame_hits[0]

            # getting the start of pixels in frame
            tile_x = tile_xy_coord[0][0]
            tile_y = tile_xy_coord[0][1]

            # defining the size of pixel  # things to do: Substitute 256 with tile col and rows
            rect_x = x1 - x0
            rect_y = y1 - y0
            if rect_x > 256:
                rect_x = 256
            if rect_y > 256:
                rect_y = 256
            frame_list = client.retrieve_instance_frames(study_instance_uid=study_id,
                                                         series_instance_uid=series_id,
                                                         sop_instance_uid=high_res_instance_ID,
                                                         frame_numbers=[hit_frame]
                                                         )
            frame = format_frame(high_res_instance, frame_list)
            best_average_pixel = np.mean(frame)
            instance_uid, instance_number, number_of_frames, pixel_spacing, stain = get_metadata_for_table(
                high_res_instance)
            line = "\t".join(
                [study_id, series_id, instance_uid, str(instance_number), str(pixel_spacing), str(number_of_frames),
                 stain, str(best_average_pixel)])
            output_file.write(line + "\n")
            img = axes_flat[n + 1].imshow(frame)
            rect = patches.Rectangle((tile_y - 2, tile_x - 2), rect_y, rect_x, linewidth=1, edgecolor='r',
                                     facecolor='none')
            axes_flat[n + 1].add_patch(rect)
            axes_flat[n + 1].set_title("Pixel_Spacing" + str(pixel_spacing))

        for a in axes_flat:
            a.set_xticklabels([])
            a.set_yticklabels([])
        # Save using bbox_inches='tight' and a small pad to crop extra white space
        fig.savefig("Series_" + str(series_count) + ".png", bbox_inches='tight', pad_inches=0.02, dpi=200)

output_file.close()
