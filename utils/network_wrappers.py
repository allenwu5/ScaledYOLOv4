"""
 Copyright (c) 2019-2020 Intel Corporation
 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at
      http://www.apache.org/licenses/LICENSE-2.0
 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""


from abc import ABC, abstractmethod


from .ie_tools import load_ie_model


class DetectorInterface(ABC):
    @abstractmethod
    def run_async(self, frames, index):
        pass

    @abstractmethod
    def wait_and_grab(self):
        pass


class Detector(DetectorInterface):
    """Wrapper class for detector"""

    def __init__(self, ie, model_path, out_blob,
                 device='CPU', ext_path='', max_num_frames=1):
        self.net = load_ie_model(ie, model_path, out_blob, device, None, ext_path, num_reqs=max_num_frames)
        self.max_num_frames = max_num_frames

    def run_async(self, frames, index):
        assert len(frames) <= self.max_num_frames
        self.shapes = []
        for i in range(len(frames)):
            self.shapes.append(frames[i].shape)
            self.net.forward_async(frames[i])

    def wait_and_grab(self):
        all_detections = []
        outputs = self.net.grab_all_async()
        for i, out in enumerate(outputs):
            all_detections.append(out)
        return all_detections

    def get_detections(self, frames, index):
        """Returns all detections on frames"""
        self.run_async(frames, index)
        return self.wait_and_grab()