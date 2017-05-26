kill $(cat p.txt)
nohup matlab <run.m >o.txt 2>e.txt&
echo $! >p.txt

