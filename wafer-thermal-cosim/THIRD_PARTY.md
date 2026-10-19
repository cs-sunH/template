# Third-party scope after thermal-core trim

活动包只需要 NumPy/SciPy 和可选 HotSpot/3D-ICE native bridge。固定版本的第三方
checkout 已放在本仓库的 `upstream/`，版本和用途由 `repos.lock.json` 记录。

Timeloop、ATLAS、BookSim、DRAMSim3、Accelergy、MFIT 及其 profile/DSE 工具属于离线工具，不会在 thermal-core import 或每个热窗启动。各 checkout 自带的源码和许可证文件保留在 `upstream/`；完整研究材料仍保留在原始备份中。
