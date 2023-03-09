import paho.mqtt.client as mqtt
import time
import sys

def on_connect(client, userdata, flags, rc):
    print(f"Connected with result code {rc}")


def main():
    client = mqtt.Client()
    client.on_connect = on_connect
    client.connect("broker.emqx.io", 1883, 60)

    client.publish(f'raspberry/sensorweb{sys.argv[1]}', payload=sys.argv[2], qos=0, retain=False)
    print(f"send {sys.argv[1]} to raspberry/topic")
    # time.sleep(1)
    quit()


if __name__ == "__main__":
    progname = sys.argv[0]
    if len(sys.argv) < 2:
        print(f"Usage: python3 {progname} <topic> <payload>")
        print(f"Example: python3 {progname} motor1 1 (send command 1 to motor 1)")
        exit()
    main()