import socket
import struct
import math, time
import torch

#for receive and send data
#also for processing data
Size_f = 4

localIP = ""
remoteIP = "10.100.193.238"
localPort = 8080
remotePort = 8080
len_data_f_skeleton = 18    #length of data sent to simulation robot
len_data_2_skeleton = 8     #length of data feedback to exoskeleton



def contains_nan(data):
    # Check if any value in the list is NaN
    return any(math.isnan(x) for x in data)

class Communicating:
    def __init__(self):

        global localIP

        localIP = socket.gethostbyname(socket.gethostname())

        print("**********************local IP: " + localIP + "********************************")

        self.cmd_data = [0.0]*len_data_2_skeleton  # 
        self.master_data = [0.0, 0.15, -0.8, 0.0, 0.0, 0.0, 0.0,
                                         0.0, -0.15, -0.8, 0.0, 0.0, 0.0, 0.0,
                                         0.0, 0.0, 1, 1] # only for simple test # modify if the message from exosklton is expanded

        #self.master_data = [0.0]*len_data_f_skeleton

        print("                                                                         ")
        print("********************** command data initialized**************************")
        
        self.UDPServerSocket = socket.socket(family=socket.AF_INET, type = socket.SOCK_DGRAM)

        self.UDPServerSocket.settimeout(5) #time out setting

        self.UDPServerSocket.bind((localIP, localPort))
        print("************************UDP server up and listening***************************")
        print("                                                                              ")
        
    def listen(self, *args, **kwargs):
        
        print("----------------------listen thread start")
        listen_format_string = f'{len_data_f_skeleton}f'

        
        while True:
            print("----------------------------listen thread in loop")
            ########## receive raw data
            try:
                listeningData =  self.UDPServerSocket.recvfrom(len_data_f_skeleton * Size_f) # will block until data received # UDP head + IP + data size
                raw_data = listeningData[0]
                clientIP = listeningData[1]
                print("------------------------ raw data received----------------------")
            except socket.timeout:
                print("--------------------- receive function timeout--------------------")
                continue
            
            ########## detect data length
            if len(raw_data) == len_data_f_skeleton * Size_f:
                try:
                    templist = list(struct.unpack(listen_format_string, raw_data))  # Convert bytes data to float list
                    if len(templist) == len_data_f_skeleton:
                        self.master_data = templist
                        print(f"-----------------------Client IP address: {clientIP}")
                        print(f"-----------------------Message from client: {self.master_data[:3]}...")  # Print first 10 values for brevity
                    else:
                        print(f"----------Warning: The received data does not match expected length. Expected {len_data_f_skeleton}, got {len(templist)}")
                        continue
                except struct.error as e:
                    print(f"---------------------------Error unpacking data: {e}")
                    continue  # Skip this iteration if unpacking fails
                
            else:
                print(f"--------------Warning: received raw_data size {len(raw_data)} does not match expected {len_data_f_skeleton * Size_f}.")
                continue  # Skip this iteration if unpacking fails
            
            print("----------------------------Message received")

        
    def send(self, *args, **kwargs):

        print("-------------------------send thread start")
        send_format_string = f'{len_data_2_skeleton}f'
        while True:
            #time.sleep(2)
            print("----------------------------send thread in loop")
#
            if not self.cmd_data:  # Check if cmd_data is empty
                print("---------------------------Warning: cmd_data is empty, skipping send.")
                continue  # Skip sending if data is empty
            
            if contains_nan(self.cmd_data):
                print("---------------------------Warning: cmd_data contains NaN values.")
                continue  # Skip sending if data has nan
    
            if not len(self.cmd_data)==len_data_2_skeleton:
                print(f"--------------------------Warning: cmd_data length mismatch. Expected {len_data_2_skeleton}, got {len(self.cmd_data)}.")
                continue  # Skip sending if data is short

            self.cmd_message = struct.pack(send_format_string,*self.cmd_data)

            try:
                SendingData = self.UDPServerSocket.sendto(self.cmd_message,(remoteIP, remotePort))
            except socket.timeout: 
                print("-------------------------send function timeout--------------------")
                continue

            
            print("----------------------------Message sent")





    