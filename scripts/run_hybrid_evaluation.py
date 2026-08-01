from pathlib import Path
import subprocess

def main():
    root = Path(__file__).resolve().parents[1]
    
    print("Đang chạy evaluation cho mô hình Hybrid...")
    subprocess.run(["python", str(root / "src" / "ml" / "evalu_hybid.py")], check=True)
    
    print(" Hoàn tất evaluation! Kết quả được lưu tại evaluation/hybrid/.")

if __name__ == "__main__":
    main()