SET RAYPATH=.;c:\radiance\lib
PATH=c:\radiance\bin;%PATH%

cd ..\output\room\gridbased

c:\radiance\bin\gensky 9 21 12 -c -B 11.1731843575419 > sky\CertainIlluminanceLevel_2000.sky
c:\radiance\bin\oconv -f sky\CertainIlluminanceLevel_2000.sky sky\groundSky.rad scene\opaque\room..opq.mat scene\opaque\room..opq.rad scene\glazing\room..glz.mat scene\glazing\room..glz.rad > room.oct
c:\radiance\bin\rtrace -aa 0.25 -ab 2 -ad 512 -ar 16 -as 128 -dc 0.25 -dj 0.0 -dp 64 -ds 0.5 -dr 0 -dt 0.5 -e error.txt -h -I -lr 4 -lw 0.05 -ss 0.0 -st 0.85 room.oct < room.pts > result\room.res
c:\radiance\bin\rcalc -e "$1=(0.265*$1+0.67*$2+0.065*$3)*179" result\room.res > result\room.ill