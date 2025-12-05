kf->predict();
age += 1;
if (time_since_update > 0) hit_streak = 0;
time_since_update += 1;
// fixme: Found a mistake, here appended to history should be kf->x instead of kf->z
auto vec_out = convert_x_to_bbox(kf->x);
history.push_back(vec_out);
return vec_out;
}
Eigen::VectorXf KalmanBoxTracker::get_state() {
    return convert_x_to_bbox(kf->x);
}
}// namespace ocsort