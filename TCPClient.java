import java.io.BufferedWriter;
import java.io.OutputStreamWriter;
import java.io.PrintWriter;
import java.net.InetAddress;
import java.net.Socket;
import java.io.*;	//I was added

//import android.util.Log;


public class TCPClient implements Runnable {
		
	public static BufferedReader br = new BufferedReader(new InputStreamReader(System.in));
	public void run() {
        try {
        	InetAddress serverAddr = InetAddress.getByName("127.0.0.1");
         //log.d("TCP", "C: Connecting...");
			System.out.println("TCP"+ " C: Connecting...");	//I was added
        	Socket socket = new Socket(serverAddr, 8080);
			OUTER:
			while(true){
				//String message = "Hello from Client android emulator";
				String message = br.readLine();
				//System.out.print("Enter the message: ");
				//message = br.readLine();
				
				try {
					//Log.d("TCP", "C: Sending: '" + message + "'");
					System.out.println("TCP"+ " C: Sending: '" + message + "'");//I was added
					PrintWriter out = new PrintWriter( new BufferedWriter( new OutputStreamWriter(socket.getOutputStream())),true); 
					out.println(message);
					//Log.d("TCP", "C: Sent.");
					System.out.println("TCP "+ "C: Sent.");	//I was added
					//Log.d("TCP", "C: Done.");
					System.out.println("TCP "+ "C: Done."); 
				} 
				catch(Exception e) {
					//Log.e("TCP", "S: Error", e);
					System.out.println("TCP " +"S: Error "+ e);	//I was added
				}
				finally {
					//socket.close();
				}
			}
        } 
		catch (Exception e) {
            //Log.e("TCP", "C: Error", e);
			System.out.println("TCP "+ "C: Error "+ e);
        }
    }
	
	public static void main(String args[]){
		TCPClient ob1 = new TCPClient();
		ob1.run();
	}
}
