import com.github.javaparser.StaticJavaParser;
import com.github.javaparser.ast.CompilationUnit;
import com.github.javaparser.ast.visitor.VoidVisitor;
import com.github.javaparser.ast.body.FieldDeclaration;
import com.github.javaparser.ast.body.MethodDeclaration;
import com.github.javaparser.ast.visitor.VoidVisitorAdapter;
import com.github.javaparser.ast.body.ClassOrInterfaceDeclaration;

// import java.lang.instrument.ClassFileTransformer;
// import java.lang.instrument.IllegalClassFormatException;
// import java.security.ProtectionDomain;

import java.io.*;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.regex.Matcher;
// import org.json.JSONObject;
import java.util.regex.Pattern;



public class ExtractKeyEntitiesInClass {

    //private static String aClassPath = "";
    private static List<String> currentClassNameList = new ArrayList<>();
    // Key Entities: className, memberName, methodName, calledMethodAndMemberNameInMethod
    // For className, memberName, methodName, extract them by using javaParser

    // get classNameList
    private static class gatAnClassByPath extends VoidVisitorAdapter<List<String>> { //ClassNameCollector，获取Java类信息
        @Override
        public void visit(ClassOrInterfaceDeclaration n, List<String> collector){
            super.visit(n, collector);
            currentClassNameList.add(n.getNameAsString());
        }
    }

    public static List<String> getClassNameListByPath(String codePath) throws FileNotFoundException{
        currentClassNameList = new ArrayList<>();
        CompilationUnit cp = StaticJavaParser.parse(new File(codePath));
        VoidVisitor<List<String>> classNameVisitor = new ExtractKeyEntitiesInClass.gatAnClassByPath();
        classNameVisitor.visit(cp, new ArrayList<>());
        return currentClassNameList;
    }
    
    //收集 all members
    public static class AllMemberNameCollector extends VoidVisitorAdapter<List<String>> {
        @Override
        public void visit(ClassOrInterfaceDeclaration n, List<String> collector){
            super.visit(n, collector);
            for(final FieldDeclaration field: n.getFields()){
                // 一行声明中，可能声明了多个成员变量，所以需要对getVariables列表进行迭代
                for(int i = 0; i < field.getVariables().size(); i++){
                    collector.add(field.getVariable(i).getNameAsString());
                }
            }
        }
    }
    public static List<String> getAllMemberNames(String codePath) throws FileNotFoundException{
        List<String> memberNames = new ArrayList<>();
        CompilationUnit cp = StaticJavaParser.parse(new File(codePath));
        VoidVisitor<List<String>> memberNameVisitor = new ExtractKeyEntitiesInClass.AllMemberNameCollector();
        memberNameVisitor.visit(cp, memberNames);
        return memberNames;
    }

    //收集 all methods
    private static class AllNameCollector extends VoidVisitorAdapter<List<String>> { //MethodNameCollector，抽取方法信息
        @Override
        public void visit(MethodDeclaration md, List<String> collector){
            super.visit(md, collector);
            collector.add(md.getNameAsString());
        }
    }
    public static List<String> getAllMethodNames(String codePath) throws FileNotFoundException{
        List<String> methodNames = new ArrayList<>();
        CompilationUnit cp = StaticJavaParser.parse(new File(codePath));
        VoidVisitor<List<String>> methodNameVisitor = new ExtractKeyEntitiesInClass.AllNameCollector();
        methodNameVisitor.visit(cp, methodNames);
        return methodNames;
    }


    // For calledMethodAndMemberNameInMethod, extract them by using regex
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
            File oldFile = new File(filePath.replace(".java", "_keyEntitise.txt"));
            if (oldFile.exists()){
                oldFile.delete();
            }
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
    //收集 class 中的 Entities
    public static Set<String> getAccessEntitiesInCodeByPath(String codePath){
        String codeText = readFromFile(new File(codePath)).toString();
        //String regex = "(\\w+\\.)|(\\.\\w+)|(\\w+\\()|([A-Z][a-z]+)";
        String regex = "([A-Za-z]+\\.)|(\\.[A-Za-z]+)|([A-Za-z]+(\\(|\\)))|([A-Z][a-z]+)";
        Pattern pattern = Pattern.compile(regex);
        Matcher matcher = pattern.matcher(codeText);
        List<String> calledMethods = new ArrayList<>();
        while (matcher.find()) {//循环查找
            //匹配上了，获取本次匹配到的内容
            String group = matcher.group();
            String getOne = group.replace(".", "").replace("(", "").replace(")", "");
            if(!calledMethods.contains(getOne) && !getOne.equalsIgnoreCase("System") && !getOne.equalsIgnoreCase("out") && !getOne.equalsIgnoreCase("println") && !getOne.equalsIgnoreCase("print") && !getOne.equalsIgnoreCase("String")){
                calledMethods.add(getOne);
            }
        }
        Set<String> calledMethodsSet = new HashSet<String>(calledMethods);
        return calledMethodsSet;
    }

    public static Set<String> getKeyEntitiesInClass(String aClassPath){
        Set<String> keyEntitiesSet = new HashSet<String>();
        try {
            List<String> className = getClassNameListByPath(aClassPath);
            List<String> memberList = getAllMemberNames(aClassPath);
            List<String> methodList = getAllMethodNames(aClassPath);
            // System.out.println("className "+className);
            // System.out.println("memberList "+memberList);
            // System.out.println("methodList "+methodList);
            Set<String> calledMethodsSet = getAccessEntitiesInCodeByPath(aClassPath);
            // System.out.println("calledMethodsSet "+calledMethodsSet);
    
            keyEntitiesSet.addAll(new HashSet<String>(className));
            keyEntitiesSet.addAll(new HashSet<String>(memberList));
            keyEntitiesSet.addAll(new HashSet<String>(methodList));
            keyEntitiesSet.addAll(calledMethodsSet);
            keyEntitiesSet.add("FormalParameter");
            keyEntitiesSet.add("ReferenceType");
            keyEntitiesSet.add("MethodInvocation");
        } catch (FileNotFoundException e) {
            // TODO Auto-generated catch block
            e.printStackTrace();
        }
        return keyEntitiesSet;
    }

    public static void main(String[] args) throws Throwable {
        // aClassPath = "/home/yqx/Downloads/ExtractNameFromJavaProject-main/src/test/envyModel/data_preprocess/A.java";
        // String keysSaveTxtPath = "/home/yqx/Downloads/ExtractNameFromJavaProject-main/src/test/envyModel/data_preprocess/keywords.txt";
        // Set<String> allKeyEntitiesInClass = getKeyEntitiesInClass(aClassPath);
        // System.out.println("allKeyEntitiesInClass "+allKeyEntitiesInClass+allKeyEntitiesInClass.size());

        // writeToTxt(keysSaveTxtPath, String.join(" ",allKeyEntitiesInClass));
        Boolean classClass = true;
        if (classClass == false){
            // For class-class folder
            String rootPath = "/home/yqx/Documents/my-FeatureEnvy-dataset/Dataset_Class_Class/";
            String labelPath = rootPath + "new_labels.txt";

            FileInputStream fin = new FileInputStream(labelPath);
            InputStreamReader reader = new InputStreamReader(fin);
            BufferedReader buffReader = new BufferedReader(reader);
            String strTmp = "";
            while((strTmp = buffReader.readLine())!=null){
                // System.out.println(strTmp);
                String classXPath = rootPath + strTmp.split("  ")[0] + "/" + strTmp.split("  ")[1] + ".java";
                String classYPath = rootPath + strTmp.split("  ")[0] + "/" + strTmp.split("  ")[2] + ".java";
                String keyEntitiesXPath = classXPath.replace(".java", "_keyEntities.txt");
                String keyEntitiesYPath = classYPath.replace(".java", "_keyEntities.txt");

                Set<String> allKeyEntitiesInClassX = getKeyEntitiesInClass(classXPath);
                Set<String> allKeyEntitiesInClassY = getKeyEntitiesInClass(classYPath);
                writeToTxt(keyEntitiesXPath, String.join(" ",allKeyEntitiesInClassX));
                writeToTxt(keyEntitiesYPath, String.join(" ",allKeyEntitiesInClassY));

                //break;
            }
            buffReader.close();
        }else{
            // For method-class folder
            String rootPath = "/home/yqx/Documents/my-FeatureEnvy-dataset/Dataset_Method_Class/";
            String labelPath = rootPath + "new_labels.txt";
            FileInputStream fin = new FileInputStream(labelPath);
            InputStreamReader reader = new InputStreamReader(fin);
            BufferedReader buffReader = new BufferedReader(reader);
            String strTmp = "";
            while((strTmp = buffReader.readLine())!=null){
                //System.out.println(strTmp);
                String methodPath = rootPath + strTmp.split("  ")[0] + "/" + strTmp.split("  ")[1] + ".java";
                String classPath = rootPath + strTmp.split("  ")[0] + "/" + strTmp.split("  ")[2] + ".java";
                String keyEntitiesXPath = methodPath.replace(".java", "_keyEntities.txt");
                String keyEntitiesYPath = classPath.replace(".java", "_keyEntities.txt");
                Set<String> allKeyEntitiesInMethod = getAccessEntitiesInCodeByPath(methodPath);
                //Set<String> allKeyEntitiesInClassX = getKeyEntitiesInClass(methodPath);
                Set<String> allKeyEntitiesInClass = getKeyEntitiesInClass(classPath);
                writeToTxt(keyEntitiesXPath, String.join(" ",allKeyEntitiesInMethod));
                writeToTxt(keyEntitiesYPath, String.join(" ",allKeyEntitiesInClass));

                //break;
            }
            buffReader.close();
        }
    } 
}
