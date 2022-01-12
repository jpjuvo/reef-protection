import numpy as np
import os
import cv2

to_gray = lambda frame : cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

def shi_tomasi_points(gray):
    feature_params = dict(
        maxCorners = 300, 
        qualityLevel = 0.2, 
        minDistance = 2, 
        blockSize = 7
    )
    return cv2.goodFeaturesToTrack(
        gray, 
        mask = None, 
        **feature_params
    )

class SceneCutDetector:
    
    def __init__(
        self, 
        min_scene_length=100, 
        keypoint_func=shi_tomasi_points,
        new_scene_threshold=70.0
    ):
        """
        Estimates optical motion between two consecutive video frames 
        and detects if new frame belongs to a different sequence.
        
        Arguments:
            min_scene_length (int) : (optional) min frames in one scene before a new 
                                     frame can belong to a new scene
            keypoint_func (callable) : (optional) method that returns keypoints when 
                                       called with a grayscale frame
            new_scene_threshold (float) : (optional) estimated xy-motion (pixels) change that is 
                                          considered to be a new scene 
        """
        self.prev_gray = None
        self.prev_pts = None
        self.min_scene_length = min_scene_length
        self.keypoint_func = keypoint_func
        self.matched_prev_pts = []
        self.matched_curr_pts = []
        self.no_match_dist = 10000
        self.x_movements = []
        self.y_movements = []
        self.new_scene_threshold = new_scene_threshold
        self.n_frames_in_scene = 0
    
    def _estimate_frame_motion(self, curr_gray):
        
        try:
            # Calculate optical flow (i.e. track feature points)
            curr_pts, status, err = cv2.calcOpticalFlowPyrLK(
                self.prev_gray, curr_gray, self.prev_pts, None)
            
            assert self.prev_pts.shape == curr_pts.shape

            # less than 3 matches is not considered a match
            assert np.sum(status) > 3
            
            # Filter only valid points
            idx = np.where(status==1)[0]
            self.prev_pts = self.prev_pts[idx]
            curr_pts = curr_pts[idx]

            #Find transformation matrix
            m, inliers = cv2.estimateAffinePartial2D(
                self.prev_pts, 
                curr_pts,
                confidence=0.98
            )

            # Extract traslation
            dx = m[0,2]
            dy = m[1,2]

            # Extract rotation angle
            da = np.arctan2(m[1,0], m[0,0])
        except:
            # return no match
            return [],[], None
        
        return self.prev_pts, curr_pts, (dx,dy,da)
    
    def _record_movement(self, diffs):
        """ 
        returns manhattan distance from diffs=(dx,dy,da), 
        If None, returns self.no_match_dist
        """
        if diffs is None:
            dx = self.no_match_dist
            dy = self.no_match_dist
        else:
            dx,dy,_ = diffs
        
        # duplicate the same movement for first frame if this is the second frame
        if len(self.x_movements) == 0:
            self.x_movements.append(dx)
            self.y_movements.append(dy)
        self.x_movements.append(dx)
        self.y_movements.append(dy)
    
    def _change_in_movement(self):
        v_x_d = self.x_movements[-1] - self.x_movements[-2]
        v_y_d = self.y_movements[-1] - self.y_movements[-2]
        return np.abs(v_x_d) + np.abs(v_y_d)
    
    def update_frame(self, frame):
        """
        Detects if new frame belongs to a new scene.
        The scene cut is estimated from sudden high movement flow change.
        
        Arguments:
            frame (ndarray rgb frame) : video frame
            
        Returns:
            is_new_scene (bool) : True if this frame is a new scene
        """
        # Extract keypoints (corners)
        curr_gray = to_gray(frame)
        curr_pts = self.keypoint_func(curr_gray)
        is_new_scene = False
    
        if self.prev_gray is not None:
            self.matched_prev_pts, self.matched_curr_pts, self.diffs = self._estimate_frame_motion(curr_gray)
            self._record_movement(self.diffs)
            
            # calculte change in movement
            movement_change = self._change_in_movement()
            
            if (movement_change > self.new_scene_threshold and
                self.n_frames_in_scene >= self.min_scene_length):
                is_new_scene = True
                self.n_frames_in_scene = 0 # reset
        
        # update prev
        self.prev_pts = curr_pts
        self.prev_gray = curr_gray
        self.n_frames_in_scene += 1
        
        return is_new_scene
        
