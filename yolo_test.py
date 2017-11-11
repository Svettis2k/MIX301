import argparse
import util
import subprocess


def execute_process(cmd, cwd):
    p = subprocess.Popen(cmd, cwd=cwd, stdout=subprocess.PIPE, universal_newlines=True)
    for stdout_line in iter(p.stdout.readline, ""):
        if len(stdout_line.rstrip()) > 0:
            yield stdout_line.rstrip()
    p.stdout.close()
    return_code = p.wait()
    if return_code == 4294967295:
        print("Couldn't connect to webcam.")
    else:
        raise subprocess.CalledProcessError(return_code, cmd)


# Define input arguments for the script
parser = argparse.ArgumentParser()
parser.add_argument("-d", "--darknet_path", type=str, help="Path to 'darknet\\x64' folder")

# Parse the arguments
args = parser.parse_args()
darknet_path = args.darknet_path

util.quit_if_file_arg_is_invalid(darknet_path, "Must provide a valid path to 'darknet\\x64' folder")

command = \
    [darknet_path + "/darknet.exe", "detector", "demo",
     darknet_path + "/data/voc.data",
     darknet_path + "/yolo-voc.cfg",
     darknet_path + "/yolo-voc.weights",
     "-c", "0"]

for path in execute_process(command, darknet_path):
    start = path.find("{")
    end = path.find("}")
    print(path[start:end + 1])
