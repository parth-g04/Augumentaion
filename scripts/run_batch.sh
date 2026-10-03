#!/bin/bash
cd ~/flux_project/Augumentaion
IN=~/person1_3dgs/city_best
OUT=~/person1_3dgs/flux_out
mkdir -p $OUT

run() {
  for N in 1 2 3 4; do
    python scripts/flux_edit.py --inp $IN/city_best${N}.png --prompt "$3" --guidance $2 --out $OUT/city_best${N}_$1.png
  done
}

run rainstorm 5.0 "make the scene a heavy downpour, many clearly visible raindrops and long bright diagonal rain streaks falling across the entire image, splashes and ripples on the wet ground, dark stormy overcast sky, glossy wet roads, realistic aerial drone photograph"

run fog 4.0 "make the scene thick fog, dense low-lying mist covering the ground and the lower parts of the buildings, heavily reduced visibility, soft diffused gray light, muted colors, realistic aerial drone photograph"

run snow 4.5 "make the scene snowing, dense falling snowflakes across the whole image, with a light layer of snow collected on rooftops, road edges, trees and fields, overcast gray light, cold muted tones, realistic aerial drone photograph"

run night 4.0 "make the scene at night, dark sky, bright street lights glowing along the roads, warm lit windows in the buildings, car headlights and red tail lights visible on the roads, realistic aerial drone photograph"

run sunny 4.0 "make the scene on a clear sunny day, bright warm sunlight with soft visible light rays, long crisp shadows, vivid colors, no visible sun disc, no lens flare, realistic aerial drone photograph"