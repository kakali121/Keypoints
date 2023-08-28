import io, pygame, rpc, serial, serial.tools.list_ports, socket, sys
import cv2
import numpy as np

# Fix Python 2.x.
try: input = raw_input
except NameError: pass

interface = rpc.rpc_network_master(slave_ip="192.168.0.22", my_ip="", port=7610)

pygame.init()
screen_w = 640
screen_h = 480
try:
    screen = pygame.display.set_mode((screen_w, screen_h), flags=pygame.RESIZABLE)
except TypeError:
    screen = pygame.display.set_mode((screen_w, screen_h))
pygame.display.set_caption("Frame Buffer")
clock = pygame.time.Clock()

# Initialize the VideoWriter object.
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output.mp4', fourcc, 20.0, (screen_w, screen_h))

def jpg_frame_buffer_cb(data):
    sys.stdout.flush()

    try:
        image = pygame.image.load(io.BytesIO(data), "jpg")
        screen.blit(pygame.transform.scale(image, (screen_w, screen_h)), (0, 0))
        pygame.display.update()
        clock.tick()

        # Save the current frame to the video file.
        frame_np = pygame.surfarray.array3d(image)
        frame_np = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)
        out.write(frame_np)
        
    except pygame.error: pass

    print(clock.get_fps())

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            out.release()  # Release the video writer.
            quit()

while(True):
    sys.stdout.flush()

    result = interface.call("jpeg_image_stream", "sensor.RGB565,sensor.QVGA")
    if result is not None:
        interface.stream_reader(jpg_frame_buffer_cb, queue_depth=8)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            out.release()  # Release the video writer.
            quit()
