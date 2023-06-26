import com.github.javaparser.StaticJavaParser;
import com.github.javaparser.ast.CompilationUnit;
import com.github.javaparser.ast.visitor.VoidVisitor;
import com.github.javaparser.ast.body.MethodDeclaration;
import com.github.javaparser.ast.visitor.VoidVisitorAdapter;

import java.io.*;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.regex.Matcher;
import org.json.JSONObject;
import java.util.regex.Pattern;


public class MgdDataSetGeneration {
    private static MethodDeclaration sampleMethod;
    private static String sampleMethodName;
    private static String sourceCode = "/home/yqx/Documents/my-FeatureEnvy-dataset/sourceCode/";
    private static String sourceMethodsPath = "/home/yqx/Documents/my-FeatureEnvy-dataset/sourceMethodsPath/";
    private static String a_b_flag = "";
    private static String methodsUnitPath = "";
    private static List<String> X_content = new ArrayList<>();
    private static List<String> Edge_content = new ArrayList<>();
    private static List<String> Label_content = new ArrayList<>();
    private static String X_path = "/home/yqx/Documents/my-FeatureEnvy-dataset/MdgC2vData/MDG_X.txt";
    private static String Edge_path = "/home/yqx/Documents/my-FeatureEnvy-dataset/MdgC2vData/MDG_Edge.txt";
    private static String Label_path = "/home/yqx/Documents/my-FeatureEnvy-dataset/MdgC2vData/MDG_Label.txt";
    private static int dataItemIndex = 0;
    private static JSONObject methodFormatIndexDict = new JSONObject();

    public static List<String> readFromFile(File src) {
        try {
            try (BufferedReader bufferedReader = new BufferedReader(new FileReader(src))) {
                List<String> lines = new ArrayList<>();
                String content;
                while((content = bufferedReader.readLine() )!=null){
                    lines.add(content);
                }
                return lines;
            }
        } catch (FileNotFoundException e) {
            e.printStackTrace();
            return null;
        } catch (IOException e) {
            e.printStackTrace();
            return null;
        }
    }

    public static void writeToTxt(String filePath, String content){
        FileWriter fw = null;
        try{
            File file = new File(filePath);
            if (!file.exists())
            {
                file.createNewFile();
            }
            fw = new FileWriter(filePath);
            BufferedWriter bw = new BufferedWriter(fw);
            bw.write(content);
            bw.close();
        }catch (Exception e){
            e.printStackTrace();
        }
        finally{
            try{
                fw.close();
            }
            catch (Exception e){
                e.printStackTrace();
            }
        }
    }

    private static class MethodNameCollector extends VoidVisitorAdapter<List<String>> { //MethodNameCollector，抽取方法信息
        @Override
        public void visit(MethodDeclaration md, List<String> collector){
            super.visit(md, collector);
            collector.add(md.getNameAsString());
        }
    }

    public static List<String> getMethodNames(String codePath) throws FileNotFoundException{
        List<String> methodNames = new ArrayList<>();
        CompilationUnit cp = StaticJavaParser.parse(new File(codePath));
        VoidVisitor<List<String>> methodNameVisitor = new MgdDataSetGeneration.MethodNameCollector();
        methodNameVisitor.visit(cp, methodNames);
        return methodNames;
    }

    private static class getAnMethodByName extends VoidVisitorAdapter<List<String>> { //MethodNameCollector，抽取方法信息
        @Override
        public void visit(MethodDeclaration md, List<String> collector){
            super.visit(md, collector);
            if (md.getNameAsString().equalsIgnoreCase(sampleMethodName)){ // 通过 MethodName 来得到这个方法的代码
                sampleMethod = md; //不能返回值，那就交给全局变量 sampleMethod
                //System.out.println("sampleMethod");
                //System.out.println(sampleMethod);
            }
        }
    }

    public static MethodDeclaration getMethodByName(String codePath, String methodName) throws FileNotFoundException{
        sampleMethodName = methodName;
        CompilationUnit cp = StaticJavaParser.parse(new File(codePath));
        VoidVisitor<List<String>> methodNameVisitor = new MgdDataSetGeneration.getAnMethodByName();
        methodNameVisitor.visit(cp, new ArrayList<>());
        return sampleMethod;
    }


    //收集 method 中的 Entitise
    public static Set<String> getAccessEntitiesInMethodByPath(String codePath){
        String codeText = readFromFile(new File(codePath)).toString();
        String regex = "(\\w+\\.)|(\\.\\w+)|(\\w+\\()";
        Pattern pattern = Pattern.compile(regex);
        Matcher matcher = pattern.matcher(codeText);
        List<String> calledMethods = new ArrayList<>();
        while (matcher.find()) {//循环查找
            //匹配上了，获取本次匹配到的内容
            String group = matcher.group();
            String getOne = group.replace(".", "").replace("(", "");
            if(!calledMethods.contains(getOne)){
                calledMethods.add(getOne);
            }
        }
        Set<String> calledMethodsSet = new HashSet<String>(calledMethods);
        return calledMethodsSet;
    }
    
    private static void getMDG(String a_b_flag, String project) throws FileNotFoundException{
        methodsUnitPath = sourceCode + project + "/" + a_b_flag + "/";
        // get all java class, collect all methods of them
        List<String> methodNameList = new ArrayList<>();
        File javaPathList = new File(methodsUnitPath);
        File[] javaPaths = javaPathList.listFiles();
        for (File javaPath : javaPaths){
            if (!javaPath.isDirectory() & javaPath.isFile() & javaPath.getName().endsWith(".java")){
                String className = javaPath.getName().split("/")[javaPath.getName().split("/").length-1];
                List<String> methodNames = getMethodNames(methodsUnitPath + className);
                //System.out.println(methodNames);
                
                for (String methodName : methodNames){
                    String methodNameFormat = project + "__" + a_b_flag + "__" + className.replace(".java", "") + "__" + methodName;
                    // get X contents , add to X
                    if (!X_content.contains(dataItemIndex + " " + methodNameFormat)){
                        X_content.add(dataItemIndex + " " + methodNameFormat);
                        methodFormatIndexDict.put(methodNameFormat, dataItemIndex);
                        dataItemIndex += 1;
                    }
                    // save method code to local folder
                    String methodCode = getMethodByName(methodsUnitPath + className, methodName).toString();
                    //System.out.println(sourceMethodsPath + methodNameFormat + ".java");
                    writeToTxt(sourceMethodsPath + methodNameFormat + ".java", methodCode);
                    methodNameList.add(methodNameFormat);
                }
            }
        }
        //System.out.println(methodNameList);

        // get call relationships between methods
        for (String methodCall : methodNameList){
            int callVal = 1;
            for (String methodCalled : methodNameList){
                if (!methodCall.split("__")[3].equalsIgnoreCase(methodCalled.split("__")[3])){
                    Set<String> methodCallSet = getAccessEntitiesInMethodByPath(sourceMethodsPath + methodCall + ".java");
                    if (methodCallSet.contains(methodCalled.split("__")[3])){
                        String callEdge = methodFormatIndexDict.getInt(methodCall) + " " + methodFormatIndexDict.getInt(methodCalled);
                        if (!Edge_content.contains(callEdge)){
                            Edge_content.add(callEdge);
                            callVal += 1;
                        }
                    }
                }
            }
            if (callVal > 1 && callVal < 5){
                callVal = 2;
            }else if (callVal >= 5){
                callVal = 3;
            }
            String label_content = methodFormatIndexDict.getInt(methodCall) + " " + callVal;
            if (!Label_content.contains(label_content)){
                Label_content.add(label_content);
            }
        }
    }

    public static void main(String[] args) throws FileNotFoundException {

        File sourceFile = new File(sourceCode);
        String[] children = sourceFile.list();
        for (String project : children) {
            System.out.println(project);

            if (!project.equalsIgnoreCase("bin")){
                // methodsUnit in /a
                a_b_flag = "a";
                getMDG(a_b_flag, project);

                // methodsUnit in /b
                a_b_flag = "b";
                getMDG(a_b_flag, project);
            }

            //break;
        }
        // deal with single node
        List<Integer> edgeNum = new ArrayList<>();
        for (String edge : Edge_content){
            String [] edge_con = edge.split(" ");
            edgeNum.add(Integer.valueOf(edge_con[0]));
            edgeNum.add(Integer.valueOf(edge_con[1]));
        }
        List<Integer> singleNum = new ArrayList<>();
        for (int i=0; i<X_content.size(); i++){
            if (!edgeNum.contains(i)){
                singleNum.add(i);
            }
        }
        for (int j=0; j<singleNum.size()-1; j++){
            Edge_content.add(singleNum.get(j) + " " + singleNum.get(j+1));
        }

        writeToTxt(X_path, String.join("\n",X_content));
        writeToTxt(Edge_path, String.join("\n",Edge_content));
        writeToTxt(Label_path, String.join("\n",Label_content));
    }
}
