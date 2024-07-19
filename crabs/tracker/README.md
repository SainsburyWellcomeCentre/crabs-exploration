# Tracker

We currently use [SORT](https://github.com/abewley/sort), an algorithm based on Kalman filtering, to track the detected crabs across frames. As stated in the SORT repository, this tracker doesn't handle occlusions or re-entering of objects, and it was developed mainly as a baseline and testbed for the development of future trackers.

The configurable parameters of the tracker are defined in `crabs-exploration/crabs/tracker/config/tracking_config.yaml`:

- `iou_threshold`: defines the minimum IOU value between a ground truth box and a detection, to consider a detection a true positive. By default, 0.1.
- `score_threshold`: defines the minimum confidence score for a detection to be considered for tracking. By default, 0.1.
- `max_age`: maximum number of frames to keep a track "alive" without associated detections. By default, 10.
- `min_hits`: minimum number of detections required to initialise a track. By default, 1.

## Evaluation

We evaluate the performance of the tracker against manually labelled ground-truth. This ground-truth consists of manually annotated bounding boxes and IDs. We use MOTA (Multiple Object Tracking Accuracy) as a metric to evaluate performance. For each frame in the manually labelled clip, we can compute MOTA as:

```
MOTA = 1 - ((FN + FP + IDs) / GT)
```

where `FN` is the number of false negatives (missed detections), `FP` is the number of false positives, `IDs` is the number of identity switches, and `GT` is the total number of ground-truth objects. The higher the MOTA value, the better the tracking performance. Note that the MOTA metric is upper-bounded by 1, and lower-bounded by -Inf. For a full video clip, we report the average MOTA across frames.

To compute the total number of false negatives (or missed detections, `FN`) at a given frame `f`, we count the number of ground-truth objects that do not match with any detection at frame `f`. A ground-truth object and a detection are considered to match if their associated boxes sufficiently overlap, that is, if their intersection-over-union (IOU) is greater than a given threshold.

To compute the total number of false positives (`FP`) at a given frame `f`, we count the number of detections that do not match with any of the ground-truth object defined at frame `f`.

A true positive (`TP`) is defined as a detection that sufficiently overlaps with a ground-truth box (with overlap measured with the `IOU` metric). To compute the number of identity switches (`IDs`) at a given frame `f` we inspect the set of true positives, and check if for each of their ground-truth IDs, the predicted ID at frame `f` matches the predicted ID at the last frame the object was detected. If the predicted IDs do not match for the same ground-truth ID, we count that as one identity switch.

This is slightly different to some MOTA definitions, which only account for identity switches between consecutive frames. It is also different from other implementations, which define an "expected" predicted ID for each ground-truth ID. This "expected" predicted ID is the predicted ID that is most often (in terms of number of frames) associated to a ground-truth ID.

## References and useful resources

- Bewley, A., Ge, Z., Ott, L., Ramos, F., & Upcroft, B. (2016, September). Simple online and realtime tracking. In 2016 IEEE international conference on image processing (ICIP) (pp. 3464-3468). IEEE. [link](https://arxiv.org/abs/1602.00763)
- Bernardin, K., & Stiefelhagen, R. (2008). Evaluating multiple object tracking performance: the clear mot metrics. EURASIP Journal on Image and Video Processing, 2008, 1-10. [link](https://link.springer.com/article/10.1155/2008/246309)
- Luiten, J., Osep, A., Dendorfer, P., Torr, P., Geiger, A., Leal-Taixé, L., & Leibe, B. (2021). Hota: A higher order metric for evaluating multi-object tracking. International journal of computer vision, 129, 548-578. [link](https://link.springer.com/article/10.1007/s11263-020-01375-2)
- [TrackEval library](https://github.com/JonathonLuiten/TrackEval)
- py-motmetrics library
- [MOTChallenge Evaluation Kit](https://github.com/dendorferpatrick/MOTChallengeEvalKit)
- Ristani, E., Solera, F., Zou, R., Cucchiara, R., & Tomasi, C. (2016, October). Performance measures and a data set for multi-target, multi-camera tracking. In European conference on computer vision (pp. 17-35). Cham: Springer International Publishing. [link](https://arxiv.org/abs/1609.01775) - might be useful for multi-camera tracking evaluation.
