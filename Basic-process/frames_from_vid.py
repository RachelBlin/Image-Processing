import cv2

def get_frames_from_vid(path_video, path_frames):
  """
  A function to get all the frames in a video

  :param path_video: The path of the video to be processed
  :param path_frames: The path of the folder where to put the frames
  """
  vidcap = cv2.VideoCapture(path_video)
  success,image = vidcap.read()
  count = 0
  while success:
    cv2.imwrite(path_frames + "frame%d.png" % count, image)     # save frame as png file
    success,image = vidcap.read()
    print("Read a new frame: ", success)
    count += 1

path_video = "/home/rblin/Documents/Databases/Cerema/GoPro/2eCarte/GOPR0410.MP4"
path_frames = "/home/rblin/Documents/Databases/Cerema/GoPro/brouillard_frames/"

get_frames_from_vid(path_video, path_frames)