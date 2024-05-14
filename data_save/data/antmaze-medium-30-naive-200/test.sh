#!/bin/bash

for file in *skill_model*.npy; do
	    mv "$file" "$(echo "$file" | sed 's/antmaze-medium-30-naive-200skill_model/skill_model/')"
    done

