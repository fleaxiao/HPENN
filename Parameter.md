## Skeleton Dimension 
### measured parameter
- length: 65mm
- width: 25mm
- height: 35mm
- dww_ii = 0.2mm
- dww_oo = 0.2mm
### inference paramter
- perimeter: (65+25)*2=180mm
- foil length: 180*6 = 1080mm ~ 1m

## Network Input Name
#### IW
- Np; Ns; hw1; hw2; dww_i_x; dww_o_x; dww_ii_x; dww_oo_x; dww_x; lcore_x1; hw; dw
#### OW
- Np; Ns; hw1; hw2; dww_i_x; dww_o_x; dww_ii_x; dww_oo_x; dww_x; lcore_x1

## Input parameter range
| Parameter Name | Max          | Min         | Standard           | Gap |
| :------------: | :----------: | :---------: | :----------------: | :--:|
| Np             | 6            | 3           | 4                  | 1   |
| Ns             | 6            | 3           | 4                  | 1   |
| hw1            | 70           | 20          | 35                 | 5   |
| hw2            | 70           | 10          | 25                 | 5   |
| dww_i_x        | 0.3          | 0.1         | 0.2                | 0.1 |
| dww_o_x        | 0.3          | 0.1         | 0.2                | 0.1 |
| dww_ii_x       | 0.3          | 0.1         | 0.2                | 0.1 |
| dww_oo_x       | 0.3          | 0.1         | 0.2                | 0.1 |
| dww_x          | 6            | 0.5         | 4.5                | 0.5 |
| lcore_x1       | 4            | 1           | 3                  | 0.5 |
| hw             | 82           | 22          | 45                 | 5   |
| dw             | 22.6         | 6           | 13.3               | 5   | 
|  
| lcs            | 80           | 20          | 32                 | 2   |
| dcs            | 40           | 6           | 11                 | 2   |
| lcore_x1       | 4            | 1           | 3                  | 0.5 |
| lcore_x2       | 6            | 3           | 3                  | 0.5 |
| lcore_y1       | 4            | 1           | 5                  | 0.5 |
| lcore_y2       | 6            | 3           | 5                  | 0.5 |

## Output parameter range (after log transfer)
| Parameter Name | Max          | Min         | 
| :------------: | :----------: | :---------: |
| loss_2D        | 1            | -0.15       |   
| loss_3D        | 0.25         | -0.05       |   

## System parameter
|  Hidden Layer | Hidden Size  | 
| :-----------: | :----------: | 
|  3            | 100          | 
|


## Gnenration Time
|  Dataset Name | Size         | Time        | Core  | SBU          |
| :-----------: | :----------: | :---------: | :----:| :---------:  |
| IW            | 5w           | 16:19:42    | 48    | 768/2        |
| OW            | 5w           | 10:41:22    | 48    | 480/2        |
|
| 3-3           | 100          | 50024       | 31    | 430          |  
| 3-4           | 100          | 51048       | 31    | 440          |
| 3-5           | 100          | 72557       | 48    | 967          |
| 3-6           | 100          | 74496       | 48    | 996          |
| 4-3           | 100          | 56049       | 31    | 483          |
| 4-4           | 100          | 59049       | 31    | 508          |
| 4-5           | 100          | 69803       | 48    | 931          |
| 4-6           | 100          | 71710       | 64    | 1275         |
| 5-3           | 100          | 72087       | 48    | 961          |
| 5-4           | 100          | 72813       | 48    | 971          |
| 5-5           | 100          | 81686       | 64    | 1452         |
| 5-6           | 100          | 90896       | 64    | 1615         |
| 6-3           | 100          | 75689       | 48    | 1009         |
| 6-4           | 100          | 71498       | 64    | 1271         |
| 6-5           | 100          | 89383       | 64    | 1589         |
| 6-6           | 100          | 92558       | 60    | 1543         |
- 2D : 1248/2 * 3/5 = 374.4 (training: 225)
- 3D : 16441 (Every 100 3D data: 16441/15 = 1096)