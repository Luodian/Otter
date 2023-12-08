import subprocess

# Commands to run
commands = ["command1", "command2", "command3"]

# Iterate over commands and execute them
for command in commands:
    # Launch tmux with each command; this will keep the command running
    subprocess.Popen(["tmux", "new-session", "-d", command])
