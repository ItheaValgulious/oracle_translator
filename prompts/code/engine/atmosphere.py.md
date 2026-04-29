# atmosphere.py

这个文件提供空环境层结使用的共享参数和辅助函数。

## 对外接口

- `AMBIENT_AIR_STRATIFICATION_DELTA`
  - 定义从高空到地面的总背景温差幅度。
- `AMBIENT_AIR_RESTORE_RATE`
  - 定义空空气向背景温度梯度回归的强度。
- `DEFAULT_EMPTY_AIR_BASE_TEMPERATURE`
  - 定义默认环境空气的中性基准温度。
- `ambient_air_temperature_for_row(height, y, base_temperature)`
  - 返回指定行在背景层结下的环境空气温度。
- `default_ambient_air_temperature_for_row(height, y)`
  - 返回使用默认环境空气基准温度时的该行环境温度。

## 依赖的对外接口

- 无额外项目内依赖。

## 主要功能

- 统一描述“近地略暖,高空略冷”的弱环境温度梯度。
- 给 CPU `thermal` 和 `motion` 共用,避免两边各自写出不一致的环境温度公式。
- 让空环境温度层结既能参与热回归,也能参与气体/空气的温度相关运动判断。
- 也给 grid/sim 这类创建空空气的路径提供一致的默认环境温度来源。
