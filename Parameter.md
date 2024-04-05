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
| IW            | 5w           | 16:19:42    | 48    | 768/12 = 64  |
| OW            | 5w           | 10:41:22    | 48    | 480/12 = 40  |
|
| 3-3           | 100          | 14355       | 24    | 96           |  
| 3-4           | 100          | 17102       | 24    | 114          |
| 3-5           | 100          | 21040       | 24    | 140          |
| 3-6           | 100          | 24364       | 24    | 162          |
| 4-3           | 100          | 16475       | 24    | 110          |
| 4-4           | 100          | 19339       | 24    | 129          |
| 4-5           | 100          | 23383       | 24    | 156          |
| 4-6           | 100          | 27330       | 24    | 182          |
| 5-3           | 100          | 19443       | 24    | 130          |
| 5-4           | 100          | 22613       | 24    | 151          |
| 5-5           | 100          | 25957       | 24    | 173          |
| 5-6           | 100          | 29311       | 24    | 195          |
| 6-3           | 100          | 22955       | 24    | 153          |
| 6-4           | 100          | 26008       | 24    | 173          |
| 6-5           | 100          | 28783       | 24    | 192          |
| 6-6           | 100          | 32243       | 24    | 215          |
- 2D : 104 * 3/5 = 62.4 (training: 38)
- 3D : 2471 (Every 100 3D data: 2471/14 = 176.5)