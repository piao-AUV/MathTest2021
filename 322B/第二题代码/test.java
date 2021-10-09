public class EqualsTest{
    public static void main(String args[]){
    BankAccount a = new BankAccount("Bob", 123456, 100.00f);
    BankAccount b = new BankAccount("Bob", 123456, 100.00f);
    if (a.equals(b))
    System.out.println("YES");
    else
    System.out.println("NO");
    }
}