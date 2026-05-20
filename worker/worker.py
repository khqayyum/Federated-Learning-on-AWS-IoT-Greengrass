import io
import os
import time
import json
import threading
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import boto3
from awsiot import mqtt_connection_builder
from awscrt import mqtt

NUM_CLASSES = 10
NUM_ROUNDS = 5

round_trigger = threading.Event()
current_round_data = {}


class LeNet5(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5, padding=2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


def load_model(state_dict):
    model = LeNet5()
    model.load_state_dict(state_dict)
    return model


def serialize_state_dict(state_dict):
    buf = io.BytesIO()
    np.savez(buf, **{k: v.cpu().numpy() for k, v in state_dict.items()})
    return buf.getvalue()


def deserialize_state_dict(data):
    npz = np.load(io.BytesIO(data))
    return OrderedDict({k: torch.from_numpy(npz[k]) for k in npz.files})


class MNISTFlatDataset(Dataset):
    def __init__(self, data_dir, labels_csv, transform=None):
        self.transform = transform
        self.samples = []
        with open(labels_csv, 'r') as f:
            next(f)
            for line in f:
                parts = line.strip().split(',')
                fpath = os.path.join(data_dir, parts[0])
                if os.path.exists(fpath):
                    self.samples.append((fpath, int(parts[2])))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        fpath, label = self.samples[idx]
        img = Image.open(fpath).convert('L')
        if self.transform:
            img = self.transform(img)
        return img, label


def train_local(model, dataloader, lr=0.005, epochs=5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for _ in range(epochs):
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)
            correct += (outputs.argmax(dim=1) == labels).sum().item()
            total += images.size(0)
    return {
        "train_loss": running_loss / total,
        "train_accuracy": correct / total,
        "num_samples": len(dataloader.dataset),
    }


def on_message_received(topic, payload, dup, qos, retain, **kwargs):
    data = json.loads(payload.decode())
    current_round_data["round_number"] = data["round_number"]
    current_round_data["num_rounds"] = data.get("num_rounds", NUM_ROUNDS)
    round_trigger.set()


def wait_for_s3_object(s3, bucket, key):
    while True:
        try:
            resp = s3.get_object(Bucket=bucket, Key=key)
            return resp["Body"].read()
        except Exception:
            time.sleep(1)


def worker_main():
    partition_id = int(os.environ["PARTITION_ID"])
    asu_id = os.environ["ASU_ID"]
    iot_endpoint = os.environ["IOT_ENDPOINT"]

    global_bucket = f"{asu_id}-global-bucket"
    local_bucket = f"{asu_id}-local-bucket"
    mqtt_topic = f"fl/{asu_id}/next-round"

    s3 = boto3.client("s3", region_name="us-west-2")

    cert_path = "/greengrass/v2/thingCert.crt"
    key_path = "/greengrass/v2/privKey.key"
    ca_path = "/greengrass/v2/rootCA.pem"

    mqtt_connection = mqtt_connection_builder.mtls_from_path(
        endpoint=iot_endpoint,
        cert_filepath=cert_path,
        pri_key_filepath=key_path,
        ca_filepath=ca_path,
        client_id=f"{asu_id}-fl-worker-{partition_id}",
        clean_session=True,
        keep_alive_secs=30,
    )

    connect_future = mqtt_connection.connect()
    connect_future.result()
    print(f"Worker {partition_id} connected to IoT Core")

    subscribe_future, _ = mqtt_connection.subscribe(
        topic=mqtt_topic,
        qos=mqtt.QoS.AT_LEAST_ONCE,
        callback=on_message_received,
    )
    subscribe_future.result()
    print(f"Worker {partition_id} subscribed to {mqtt_topic}")

    data_dir = f"/home/ubuntu/fl-client/data_cache/client-{partition_id}"
    labels_csv = "/home/ubuntu/fl-client/data_cache/labels.csv"

    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    dataset = MNISTFlatDataset(data_dir, labels_csv, transform=transform)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=2)

    print(f"Worker {partition_id} waiting for MQTT trigger...")
    round_trigger.wait()
    round_trigger.clear()
    print(f"Worker {partition_id} received round 0 trigger")

    for round_id in range(NUM_ROUNDS):
        print(f"Worker {partition_id} starting round {round_id}")

        global_key = f"models/global_model_round_{round_id}.npz"
        model_data = wait_for_s3_object(s3, global_bucket, global_key)

        state_dict = deserialize_state_dict(model_data)
        model = load_model(state_dict)
        metrics = train_local(model, dataloader, lr=0.005, epochs=5)

        local_data = serialize_state_dict(model.state_dict())
        local_key = f"updates/local_model_round_{round_id}_worker_{partition_id}.npz"
        s3.put_object(Bucket=local_bucket, Key=local_key, Body=local_data)

        print(f"Worker {partition_id} round {round_id} done | loss={metrics['train_loss']:.4f} | acc={metrics['train_accuracy']:.4f}")

        if round_id < NUM_ROUNDS - 1:
            next_global_key = f"models/global_model_round_{round_id + 1}.npz"
            print(f"Worker {partition_id} waiting for {next_global_key}...")
            wait_for_s3_object(s3, global_bucket, next_global_key)
            print(f"Worker {partition_id} starting round {round_id + 1}")

    print(f"Worker {partition_id} completed all {NUM_ROUNDS} rounds.")
    mqtt_connection.disconnect().result()


if __name__ == "__main__":
    worker_main()
