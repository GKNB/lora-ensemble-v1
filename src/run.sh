#!/bin/bash

PS3='Please select a model to test: '

options=("Llama2" "Llama3" "Mistral" "Quit")

select opt in "${options[@]}"
do
    case $opt in
        "Llama2"|"Llama3"|"Mistral")
            echo "You chose to run $opt."

            # srun -p debug -N 1 -J test \
            srun -p csi -q csi -A csigeneral -t 24:00:00 -N 1 --gres=gpu:2 -J LLM \
            python3 /sdcc/u/rengel/src/run_model.py \
            --model_name $opt
            
            break
            ;;
        "Quit")
            echo "Exiting."
            break
            ;;
        *)
            echo "Invalid option $REPLY. Please try again."
            ;;
    esac
done
