import os
from pprint import pprint
import json
from PIL import Image, ImageDraw, ImageFont

measurements_file = "/media/grant/FastData/carla_saves/videos/measurements-0.2.2.json"
images_folder = "/media/grant/FastData/carla_saves/videos/measurements-images/"

# Read in the measurements
with open(measurements_file) as f:
    measurements = json.load(f)

# Make the folder if needed
if not os.path.exists(images_folder):
    os.mkdir(images_folder)

class MeasurementRow:
    def __init__(self, data):
        action = data["action"]
        self.throttle = action[0]
        self.turn = action[1]
        self.collision_value = data["collision_other"] + data["collision_pedestrians"] + data["collision_vehicles"]
        self.is_collision = self.collision_value > 0
        self.speed = data["forward_speed"] * 3.6
        self.offroad = data["intersection_offroad"]
        self.otherlane = data["intersection_otherlane"]
        self.step = data["step"]
        self.reward_step = data["reward"]
        self.x = data["x"]
        self.y = data["y"]
        self.reward_total = data["total_reward"]
        self.filename = "measurement_step_{:>04}.png".format(self.step)

class DrawWrapper:
    def __init__(self, img):
        self.d = ImageDraw.Draw(img)
        self.x = 10
        self.y = 10
        self.line = 18
        # self.fnt = ImageFont.truetype('/Library/Fonts/Arial.ttf', 12)
        self.font = ImageFont.truetype("FreeMono.ttf", 18)

    def text(self, header, text):
        self.d.text((self.x, self.y), header, font=self.font, fill=(0, 255, 255))
        fill = " " * len(header)
        self.d.text((self.x, self.y), fill + text, font=self.font, fill=(255, 255, 255))
        self.next_line()

    def next_line(self):
        self.y += self.line

    def shift_right(self, amount):
        self.x += amount

# Iterate all measurements
for m in measurements:
    row = MeasurementRow(m)
    img = Image.new('RGB', (456, 228), color=(0, 0, 0))
    draw = DrawWrapper(img)

    # Draw!
    draw.text("Step: ", "{}".format(row.step))
    draw.text("Reward - Step: ", "{:.3f}".format(row.reward_step))
    draw.text("Reward - Total: ", "{:>04.1f}".format(row.reward_total))

    draw.next_line()
    draw.text("Speed: ", "{:.2f}km/hr".format(row.speed))
    draw.text("Position: ", "({:>03.2f}, {:>03.2f})".format(row.x, row.y))
    draw.text("Collision: ", "{:b}".format(row.is_collision))
    draw.text("Offroad: ", "{:>03}%".format(int(row.offroad * 100)))
    draw.text("Otherlane: ", "{:>03}%".format(int(row.otherlane * 100)))

    draw.next_line()
    draw.text("Action - Throttle: ", "{:.3f}%".format(row.throttle))
    draw.text("Action - Steering: ", "{:.3f}%".format(row.turn))

    # Save
    img.save(images_folder + row.filename)

