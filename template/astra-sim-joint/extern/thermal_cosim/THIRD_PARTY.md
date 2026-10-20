# Third-party scope after thermal-core trim

活动包只需要 NumPy/SciPy 和可选 HotSpot/3D-ICE native bridge。固定版本的第三方
checkout 已放在本仓库的 `upstream/`，版本和用途由 `repos.lock.json` 记录。

Timeloop、ATLAS、BookSim、DRAMSim3、Accelergy、MFIT 及其 profile/DSE 工具属于离线工具，不会在 thermal-core import 或每个热窗启动。各 checkout 自带的源码和许可证文件保留在 `upstream/`；完整研究材料仍保留在原始备份中。

`configs/energy/` 保留静态动作能量表及其来源记录，不包含在线性能服务模型。EnergyModel 使用时由 host 显式提供硬件事实；energy-table 中的容量、macro 组织和工具输出属于适用域/来源依据，不代替当前 host 配置。被裁掉的 `service-v1` 性能表没有活动消费者，因此不再随活动输入分发。
