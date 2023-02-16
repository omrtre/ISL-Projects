import java.util.Scanner;
public class ShoutAndWhisper
{ 
public static void main (String[] args)
{
    Scanner feedback = new Scanner(System.in);
    System.out.println("Enter the text: ");
    String input=feedback.nextLine();
    System.out.println("The text in all upper case is: ");
    System.out.println(input.toUpperCase());
    System.out.println("The text in all lower case is: ");
    System.out.println(input.toLowerCase());
}
}
