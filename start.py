import subprocess
from threading import Thread

# Конфигурация
username = "ubuntu"
hosts = ["158.160.184.70", "158.160.177.21", "158.160.181.245"]
master_port = 29501
training_script = "/home/ubuntu/htohto.py"

def launch_on_host(host, rank):
    cmd = (
        "source /home/ubuntu/my_env/bin/activate && "
        f"torchrun --nnodes={len(hosts)} --nproc_per_node=1 --node_rank={rank} "
        f"--master_addr={hosts[0]} --master_port={master_port} "
        f"{training_script} "
        f"--world_size {len(hosts)} --rank {rank} --master_addr {hosts[0]} --master_port {master_port} "
        "--data_dir /home/ubuntu/mounter/image_andrew_hurtle "
        "--val_data_dir /home/ubuntu/mounter/image_andrew_hurtle "
        "--save_dir /home/ubuntu/mounter/result"
    )
    print(f"Запуск на {host} с командой: {cmd}")  # Для отладки
    subprocess.run(f'ssh {username}@{host} "{cmd}"', shell=True)

if __name__ == "__main__":
    print(f"Запуск на {len(hosts)} машинах")
    threads = [Thread(target=launch_on_host, args=(host, i)) 
              for i, host in enumerate(hosts)]
    for t in threads: t.start()
    for t in threads: t.join()
    print("Обучение завершено")