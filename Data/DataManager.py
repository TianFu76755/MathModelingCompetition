import pandas as pd
from pathlib import Path


class DataManager:
    def __init__(self):
        """
        自动回溯到 MathModelingCompetition/Data 目录
        """
        # 从当前文件所在路径开始向上查找
        current = Path(__file__).resolve()
        found = False
        for parent in current.parents:
            if parent.name == "MathModelingCompetition":
                self.base_dir = parent
                found = True
                break

        if not found:
            raise FileNotFoundError("未找到 MathModelingCompetition 目录，请检查项目结构。")

        self.file_dir = self.base_dir / "Data"
        if not self.file_dir.exists():
            raise FileNotFoundError(f"未找到 Data 目录: {self.file_dir}")
        self.dfs = {}

    def load_all(self):
        for i in range(1, 5):
            file_path = self.file_dir / f"附件{i}.xlsx"
            if not file_path.exists():
                raise FileNotFoundError(f"缺少文件: {file_path}")

            df = pd.read_excel(file_path, sheet_name=0)  # 只有一个sheet
            # 统一列名（去除空格、换行）
            df.columns = [col.strip() for col in df.columns]
            self.dfs[f"附件{i}"] = df
        return self.dfs

    def get_data(self, idx):
        """
        根据索引获取指定附件的数据
        idx: 1~4
        """
        key = f"附件{idx}"
        if key not in self.dfs:
            raise ValueError
        return self.dfs[key]


DM: DataManager = DataManager()
DM.load_all()


if __name__ == "__main__":
    all_data = DM.load_all()

    # 打印附件1的数据前5行
    print(DM.get_data(1).head())

    # 遍历所有数据
    for name, df in all_data.items():
        print(f"{name} 共有 {len(df)} 行数据")
