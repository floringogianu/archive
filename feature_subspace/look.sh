#!/bin/bash

echo "EXPERIMENT:  $1"

if [ "$2" = "--do" ]; then
    echo "Will delete files..."
else
    echo "Just a simulation; use --do to delete files for real."
fi

((locked = 0))
((new = 0))
((old = 0))
((done = 0))
((dead = 0))

for var_name in $(ls $1); do
    var_path="$1/$var_name"
    if [ -d $var_path ]; then
        for run_id in $(ls $var_path); do
            run_path="$var_path/$run_id"
            if [ -d $run_path ]; then

                END_PATH="$run_path/.__end"
                LOCK_PATH="$run_path/.__lock"
                START_PATH="$run_path/.__start"
                TRACE_PATH="$run_path/trace.th"
                OUT_PATH="$run_path/out"

                if [ -f "$LOCK_PATH" ]; then # It's locked
                    ((locked = locked + 1))
                    ((crt_time = $(date +"%s")))
                    ((last_touched = $(stat -c %Y "$OUT_PATH")))
                    ((elapsed = crt_time - last_touched)) # How long ago it was locked

                    if [ "$elapsed" -gt 3600 ]; then # 24 hours
                        ((old = old + 1))
                        if [ -f "$END_PATH" ]; then
                            if [ "$2" = "--do" ]; then
                                rm "$LOCK_PATH"
                            fi
                            ((done = done + 1))
                        else
                            if [ "$2" = "--do" ]; then
                                rm -f "$START_PATH" "$LOCK_PATH" "$END_PATH"
                            fi
                            ((dead = dead + 1))
                        fi
                        echo "$var_name/$run_id : $elapsed"
                    else
                        ((new = new + 1))
                    fi
                fi
            fi
        done
    fi
done

echo "Locked: $locked; New: $new + Old: $old (Ended: $done + Dead: $dead)"
