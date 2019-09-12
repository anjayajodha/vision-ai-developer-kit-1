# Copyright (c) 2018, The Linux Foundation. All rights reserved.
# Licensed under the BSD License 2.0 license. See LICENSE file in the project root for
# full license information.

import argparse
import sys
import time
import subprocess
import utility
import os
import iot

import cv2
from azure.storage.blob import BlockBlobService, PublicAccess
from azure.storage.queue import QueueService
import VideoStream
from VideoStream import VideoStream

from camera import CameraClient

def main(protocol=None):
    print("\nPython %s\n" % sys.version)
    parser = argparse.ArgumentParser()
    parser.add_argument('--ip', help='ip address of the camera', default=utility.getWlanIp())
    parser.add_argument('--username', help='username of the camera', default='admin')
    parser.add_argument('--password', help='password of the camera', default='admin')
    args = parser.parse_args()
    ip_addr = args.ip
    username = args.username
    password = args.password
    #ip_addr = '127.0.0.1'
    hub_manager = iot.HubManager()

    STORAGE_ACCOUNT_NAME = os.environ.get('STORAGE_ACCOUNT_NAME')
    STORAGE_ACCOUNT_KEY = os.environ.get('STORAGE_ACCOUNT_KEY')
    STORAGE_ACCOUNT_SUFFIX = os.environ.get('STORAGE_ACCOUNT_SUFFIX')

    block_blob_service = BlockBlobService(account_name=STORAGE_ACCOUNT_NAME, account_key=STORAGE_ACCOUNT_KEY, endpoint_suffix=STORAGE_ACCOUNT_SUFFIX)

    container_name = 'fromcamera'

    block_blob_service.create_container(container_name)

    block_blob_service.set_container_acl(container_name, public_access=PublicAccess.Container)

    with CameraClient.connect(ip_address=ip_addr, username=username, password=password) as camera_client:
        #transferring model files to device
        utility.transferdlc()

        print(camera_client.configure_preview(display_out=1))

        camera_client.toggle_preview(True)
        time.sleep(15)
        rtsp_stream_addr = "rtsp://" + ip_addr + ":8900/live"
        hub_manager.iothub_client_sample_run(rtsp_stream_addr)

        camera_client.toggle_vam(True)

        camera_client.configure_overlay("inference")

        camera_client.toggle_overlay(True)

        capture = VideoStream(rtsp_stream_addr).start()
        try:
            with camera_client.get_inferences() as results:
                print_inferences(hub_manager,capture, block_blob_service,results)
        except KeyboardInterrupt:
            print("Stopping")
        try:
            while(True):
                time.sleep(2)
        except KeyboardInterrupt:
            print("Stopping")

        #camera_client.toggle_overlay(False)

        camera_client.toggle_vam(False)

        camera_client.toggle_preview(False)

def get_model_config():
    # TODO: get the AML model and return an AiModelConfig
    return None

def print_inferences(hub_manager,capture, block_blob_service,results=None):
    print("")
    for result in results:
        if result is not None and result.objects is not None and len(result.objects):
            timestamp = result.timestamp
            #if timestamp:
                #print("timestamp={}".format(timestamp))
            #else:
                #print("timestamp= " + "None")
            for object in result.objects:
                id = object.id
                label = object.label
                confidence = object.confidence
                x = object.position.x
                y = object.position.y
                w = object.position.width
                h = object.position.height
                print("id={}".format(id))
                print("label={}".format(label))
                print("confidence={}".format(confidence))
                print("Position(x,y,w,h)=({},{},{},{})".format(x, y, w, h))
                print("")
                # hub_manager.SendMsgToCloud("I see " + str(label) + " with confidence :: " + str(confidence))
                # time.sleep(1)
            if any(obj.label == 'person' for obj in result.objects):
                start=time.time()
                print(">> Capturing image: " + str(start))
                frame = capture.read()
                #Flip color space if necessary
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                jp = cv2.imencode('.jpg',rgb_frame)[1].tostring()
                blob_name=str(start) +'.jpg'
                blobprops = block_blob_service.create_blob_from_bytes(container_name, blob_name, jp)
                # Dimensions to crop image
                save_end = time.time()
                print("Time - Capture to Save: " + str((save_end-start)))
                print("<< Completed. Start time: " + str(start))
        time.sleep(1.0)
        else:
            print("No results")

if __name__ == '__main__':
    main()