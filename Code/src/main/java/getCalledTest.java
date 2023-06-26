import java.io.IOException;
import java.io.*;
import java.util.ArrayList;
import java.util.List;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

public class getCalledTest {

    public static String readFromFile(File src) {
        try {
            try (BufferedReader bufferedReader = new BufferedReader(new FileReader(src))) {
                List<String> lines = new ArrayList<>();
                String content;
                while((content = bufferedReader.readLine() )!=null){
                    lines.add(content);
                }
                return lines.toString();
            }
        } catch (FileNotFoundException e) {
            e.printStackTrace();
            return null;
        } catch (IOException e) {
            e.printStackTrace();
            return null;
        }
    }

    public static List<String> getAccessEntities(String codePath){
        String codeText = readFromFile(new File(codePath));
        String regex = "(\\w+\\.)|(\\.\\w+)|(\\w+\\()";
        Pattern pattern = Pattern.compile(regex);
        Matcher matcher = pattern.matcher(codeText);
        List<String> CalledMethods = new ArrayList<>();
        
        while (matcher.find()) {//循环查找
            //匹配上了，获取本次匹配到的内容
            String group = matcher.group();
            String getOne = group.replace(".", "").replace("(", "");
            if(!CalledMethods.contains(getOne)){
                CalledMethods.add(getOne);
            }
        }
        return CalledMethods;
    }

    public static void main(String[] args){
        String codePath = "src/test/envyModel/data_preprocess/A.java";
		System.out.println(getAccessEntities(codePath));
    }
}
