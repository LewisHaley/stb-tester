"""
Copyright © 2014-2022 Stb-tester.com Ltd.
Copyright © 2014 YouView TV Ltd.

License: LGPL v2.1 or (at your option) any later version (see
https://github.com/stb-tester/stb-tester/blob/master/LICENSE for details).
"""

from __future__ import annotations

import warnings
from typing import Optional

import cv2

from .config import get_config
from .imgutils import (
    crop, Frame, _frame_repr, _image_region, pixel_bounding_box)
from .logging import debug, draw_source_region, ImageLogger
from .mask import load_mask, MaskTypes
from .types import Region


def is_screen_black(frame: Optional[Frame] = None,
                    mask: MaskTypes = Region.ALL,
                    threshold: Optional[int] = None,
                    region: Region = Region.ALL) -> _IsScreenBlackResult:
    """Check for the presence of a black screen in a video frame.

    :param Frame frame:
      If this is specified it is used as the video frame to check; otherwise a
      new frame is grabbed from the device-under-test. This is an image in
      OpenCV format (for example as returned by `frames` and `get_frame`).

    :param str|numpy.ndarray|Mask|Region mask:
        A `Region` or a mask that specifies which parts of the image to
        analyse. This accepts anything that can be converted to a Mask using
        `stbt.load_mask`. See :doc:`masks`.

    :param int threshold:
      Even when a video frame appears to be black, the intensity of its pixels
      is not always 0. To differentiate almost-black from non-black pixels, a
      binary threshold is applied to the frame. The ``threshold`` value is in
      the range 0 (black) to 255 (white). The global default (20) can be
      changed by setting ``threshold`` in the ``[is_screen_black]`` section of
      :ref:`.stbt.conf`.

    :param Region region:
      Deprecated synonym for ``mask``. Use ``mask`` instead.

    :returns:
        An object that will evaluate to true if the frame was black, or false
        if not black. The object has the following attributes:

        * **black** (*bool*) – True if the frame was black.
        * **frame** (`stbt.Frame`) – The video frame that was analysed.

    Changed in v33: ``mask`` accepts anything that can be converted to a Mask
    using `load_mask`. The ``region`` parameter is deprecated; pass your
    `Region` to ``mask`` instead. You can't specify ``mask`` and ``region``
    at the same time.
    """
    if threshold is None:
        threshold = get_config('is_screen_black', 'threshold', type_=int)

    if frame is None:
        from stbt_core import get_frame
        frame = get_frame()

    if region is not Region.ALL:
        if mask is not Region.ALL:
            raise ValueError("Cannot specify mask and region at the same time")
        warnings.warn(
            "stbt.is_screen_black: The 'region' parameter is deprecated; "
            "pass your Region to 'mask' instead",
            DeprecationWarning, stacklevel=2)
        mask = region

    mask_, region = load_mask(mask).to_array(_image_region(frame))

    draw_source_region(frame, region)
    imglog = ImageLogger("is_screen_black", region=region, threshold=threshold)
    imglog.imwrite("frame", frame)

    grayframe = cv2.cvtColor(crop(frame, region), cv2.COLOR_BGR2GRAY)
    if mask_ is not None:
        imglog.imwrite(
            "mask", mask_,
            description=(
                "Mask applied to the image due to the 'mask' parameter "
                f"({mask}).  This is a binary image where white pixels "
                "indicate the parts of the frame that were analysed for "
                "blackness, and black pixels indicate the parts of the frame "
                "that were ignored."
            ),
            source_region=region,
        )
        cv2.bitwise_and(grayframe, mask_, dst=grayframe)
    maxVal = grayframe.max()

    result = _IsScreenBlackResult(bool(maxVal <= threshold), frame)
    debug("is_screen_black: {found} black screen using mask={mask}, "
          "threshold={threshold}: {result}, maximum_intensity={maxVal}".format(
              found="Found" if result.black else "Didn't find",
              mask=mask,
              threshold=threshold,
              result=result,
              maxVal=maxVal))

    if imglog.enabled:
        mask_desc = " and applying the mask" if mask_ is not None else ""
        imglog.imwrite(
            "gray", grayframe, description=(
                "Grayscale version of the relevant portion of the "
                "source frame after cropping to the region of interest "
                f"({region}){mask_desc}.  The values "
                "here are in the range 0 (black) to 255 (white) and will "
                "be compared against the threshold "
                f"({threshold}) to determine whether the screen is black. "
                f"The maximum pixel intensity in this image is {maxVal}."
            ),
            source_region=region,
        )
        _, thresholded = cv2.threshold(grayframe, threshold, 255,
                                       cv2.THRESH_BINARY)
        if result.black:
            bounding_box = None
        else:
            orig_bounding_box = pixel_bounding_box(thresholded)
            assert orig_bounding_box
            bounding_box = orig_bounding_box.translate(region)
        imglog.imwrite(
            "non_black", thresholded, description=(
                "Binary image showing the pixels that were considered "
                "non-black after applying the threshold. The white pixels in "
                "this image indicate the pixels that were considered non-black "
                "and the black pixels indicate the pixels that were considered "
                "black.  This corresponds to the region {region} of the source "
                "frame."
            ),
            source_region=region,
        )
        imglog.set(maxVal=maxVal, non_black_region=bounding_box)
    _log_image_debug(imglog, result)

    return result


class _IsScreenBlackResult():
    def __init__(self, black: bool, frame: Frame):
        self.black: bool = black
        self.frame: Frame = frame

    def __bool__(self):
        return self.black

    def __repr__(self):
        return ("_IsScreenBlackResult(black=%r, frame=%s)" % (
            self.black,
            _frame_repr(self.frame)))


def _log_image_debug(imglog: ImageLogger, result):
    if not imglog.enabled:
        return

    template = """\
        <h4>is_screen_black: {{result.black}}</h4>

        {{ annotated_image(non_black_region) }}

        {% if "mask" in images %}
        <h5>Mask:</h5>
        {{ img("mask") }}
        {% endif %}

        <h5>Grayscale, masked:</h5>
        {{ img("gray") }}
        <ul>
          <li>Maximum pixel intensity: {{maxVal}}
          <li>threshold={{threshold}}
        </ul>

        {% if not result.black %}
        <h5>Non-black pixels in region:</h5>
        {{ img("non_black") }}
        {% endif %}
    """

    imglog.html(template, result=result)
