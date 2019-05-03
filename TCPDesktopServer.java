import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.net.ServerSocket;
import java.net.Socket;
import java.io.*;


public class TCPDesktopServer implements Runnable{

    public static final String SERVERIP = "127.0.0.1"; // My system's IP Address
    public static final int SERVERPORT = 8080;

    public void run() {
         try {
        	 System.out.println("S: Connecting...");
             ServerSocket serverSocket = new ServerSocket(SERVERPORT);
			 
			 Socket client = serverSocket.accept();
           	 System.out.println("S: Receiving...");
			 OUTER:
             while (true) {      	 
            	 try {

                      BufferedReader in = new BufferedReader(new InputStreamReader(client.getInputStream()));
                      String str = in.readLine();

                      Process p = Runtime.getRuntime().exec("python accel_predict.py "+str);
                      BufferedReader p_in = new BufferedReader(new InputStreamReader(p.getInputStream()));
                      String ret = p_in.readLine();

                      System.out.println("S: CURRENT ACTIVITY$ " + ret);
                    
            	 } catch(Exception e) {
                        System.out.println("S: Error");
                        e.printStackTrace();
            	 } finally {
                    	//client.close();
                        System.out.println("S: Done.");
					  }
			 }
                   
         } catch (Exception e) {
             System.out.println("S: Error");
             e.printStackTrace();
         }
    }

    public static void main (String a[]) {

    	Thread desktopServerThread = new Thread(new TCPDesktopServer());
    	desktopServerThread.start();
    }
}