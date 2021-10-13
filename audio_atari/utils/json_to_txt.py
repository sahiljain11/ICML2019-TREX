import os
import argparse
import json

"""
python json_to_txt.py --json_dir ./frames/trajectories_json --output_dir ./frames/trajectories --screen_dir ./frames/screens/mspacman
"""

def load_json_data(file_path: str) -> dict:
    print(f"File Path: {file_path}")

    # first, we need to go into the file. json package automatically converts it!
    f = open(file_path, "r")
    json_data = json.loads(json.load(f))
    return json_data


def write_txt(json_data: dict, output_file: str, db_name: str) -> None:
    f = open(output_file, "w")
    f.write(f"db traj id : {db_name}\n")

    trajectory = json_data["trajectory"]
    for timestep in trajectory.keys():
        data = trajectory[timestep]
        if abs(data['reward']) > 200:
            data['reward'] = 0
            data['score'] = 0

        data['terminal'] = 1 if data['terminal'] else 0

        line = f"{timestep}, {data['reward']}, {data['score']}, {data['terminal']}, {data['action']}\n"
        f.write(line)
    return

def rename_screens(screen_dir: str) -> None:
    print(screen_dir)
    for image in os.listdir(screen_dir):
        splitted = image.split("_")
        if len(splitted) < 4:
            break
        timestep_dot_png = image.split("_")[3]

        os.rename(f"{screen_dir}/{image}", f"{screen_dir}/{timestep_dot_png}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--json_dir',   default='', help="JSON directory that needs to be translated into a txt file")
    parser.add_argument('--output_dir', default='', help="Output directory where txt files will be placed")
    parser.add_argument('--screen_dir', default='', help="Screen directory to be renamed to 1.png, 2.png, etc.")
    args = parser.parse_args()

    if args.json_dir == '' or not os.path.exists(args.json_dir):
        raise Exception("Please specify valid directory to convert")

    files = os.listdir(args.json_dir)
    for item in files:
        name = item.split(".")[0]
        json_data = load_json_data(f"{args.json_dir}/{item}")

        write_txt(json_data, f"{args.output_dir}/{name}.txt", name)
        rename_screens(f"{args.screen_dir}/{name}")