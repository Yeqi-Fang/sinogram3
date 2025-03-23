#!/bin/bash

# Create a script file in your home directory
cat > ~/monitor_and_run.sh << 'EOF'
#!/bin/bash

PID_TO_MONITOR=334527
COMMAND="python sinogram3/main.py --data_dir /mnt/d/fyq/sinogram/
2e9div_smooth --mode train --batch_size 24 --num_epochs 15 --models_dir checkpoints --attention 1 --lr 1e-4 --transformer 1 --log_dir ~/sinogram3/log"

echo "Starting monitor for process $PID_TO_MONITOR"
echo "Will run: $COMMAND"

# Monitor the process using ps which doesn't require root
while ps -p $PID_TO_MONITOR > /dev/null 2>&1; do
    echo "Process $PID_TO_MONITOR is still running. Checking again in 60 seconds..."
    sleep 60
done

echo "Process $PID_TO_MONITOR has terminated. Starting the command..."

# Log date and time to user's directory
echo "Command started at $(date)" >> ~/command_log.txt

# Run the command in background with output logged to user's home directory
cd $(dirname $(readlink -f $0))  # Change to the script's directory
nohup $COMMAND > ~/command_output.log 2>&1 &

# Get the PID of the new process
NEW_PID=$!
echo "Command started with PID: $NEW_PID"
echo "Command running with PID: $NEW_PID" >> ~/command_log.txt
echo "Check ~/command_output.log for program output"
EOF

# Make it executable
chmod +x ~/monitor_and_run.sh

# Start the monitoring script in the background (no root needed)
nohup ~/monitor_and_run.sh > ~/monitor.log 2>&1 &

echo "Monitoring script started in background. Check ~/monitor.log for status."
echo "You can use 'ps aux | grep monitor_and_run.sh' to verify it's running."