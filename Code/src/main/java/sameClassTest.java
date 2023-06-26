import com.github.javaparser.StaticJavaParser;
import com.github.javaparser.ast.CompilationUnit;
import com.github.javaparser.ast.visitor.VoidVisitor;
import com.github.javaparser.ast.body.MethodDeclaration;
import com.github.javaparser.ast.visitor.VoidVisitorAdapter;
import com.github.javaparser.ast.body.ClassOrInterfaceDeclaration;

import java.io.*;
import java.util.ArrayList;
import java.util.List;

public class sameClassTest {
   
    //全局变量
    private static ClassOrInterfaceDeclaration sampleClass;
    private static MethodDeclaration sampleMethod;
    private static String sampleMethodName;
    //private static String needRefactmethodName;
    //原始数据集路径
    private static String rootPath = "/home/yqx/Documents/my-FeatureEnvy-dataset/SourceCode";  //存储待解析Java文件的根目录
    private static String csvPath = "/home/yqx/Documents/my-FeatureEnvy-dataset/groundTruth.csv";
    //数据增强后的数据集路径
    //private static String dataset_Class_Class_Path = "/home/yqx/Documents/my-FeatureEnvy-dataset/Dataset_Class_Class";
    private static String dataset_Method_Class_Path = "/home/yqx/Documents/my-FeatureEnvy-dataset/Dataset_Method_Class";
    //数据条目保存路径
    private static String dataItemPath = dataset_Method_Class_Path+'/'+"dataItem"+'/';
    //统计正负样本数量
    //private static int posSamples = 0;
    //private static int negSamples = 0;

    private static class MethodNameCollector extends VoidVisitorAdapter<List<String>> { //MethodNameCollector，抽取方法信息
        @Override
        public void visit(MethodDeclaration md, List<String> collector){
            super.visit(md, collector);
            collector.add(md.getNameAsString());
        }
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

    private static class ClassNameCollector extends VoidVisitorAdapter<List<String>> { //ClassNameCollector，获取Java类信息

        @Override
        public void visit(ClassOrInterfaceDeclaration n, List<String> methodRemoveList){
            super.visit(n, methodRemoveList);
            try {
                for(int i = 0; i < n.getMembers().size(); i++){
                    for(String name: methodRemoveList){
                        //System.out.println(name);
                        for(int j = 0; j < n.getMethodsByName(name).size(); j++){
                            if(n.getMethodsByName(name).get(j) == n.getMembers().get(i)){
                                //System.out.println("remove  "+name);
                                n.getMembers().remove(i);
                            }
                        }
                    }
                }
    
                if(n.isTopLevelType()){ //保存每个类,但是只要外部类
                    sampleClass = n; //不能返回值，那就交给全局变量 newClass
                }
            } catch (Exception e) {
                //TODO: handle exception

                System.out.println("exception_________exception");
            }
            
        }
    }

    private static class gatAnClassByPath extends VoidVisitorAdapter<List<String>> { //ClassNameCollector，获取Java类信息
        @Override
        public void visit(ClassOrInterfaceDeclaration n, List<String> methodRemoveList){
            super.visit(n, methodRemoveList);
            try {
                if(n.isTopLevelType()){ //保存每个类,但是只要外部类
                    sampleClass = n; //不能返回值，那就交给全局变量 newClass
                }
            } catch (Exception e) {
                //TODO: handle exception
                System.out.println("exception_________exception");
            }
            
        }
    }

    public static List<String> getMethodNames(String codePath) throws FileNotFoundException{
        List<String> methodNames = new ArrayList<>();
        CompilationUnit cp = StaticJavaParser.parse(new File(codePath));
        VoidVisitor<List<String>> methodNameVisitor = new sameClassTest.MethodNameCollector();
        methodNameVisitor.visit(cp, methodNames);
        return methodNames;
    }

    public static MethodDeclaration getMethodByName(String codePath, String methodName) throws FileNotFoundException{
        sampleMethodName = methodName;
        CompilationUnit cp = StaticJavaParser.parse(new File(codePath));
        VoidVisitor<List<String>> methodNameVisitor = new sameClassTest.getAnMethodByName();
        methodNameVisitor.visit(cp, new ArrayList<>());
        return sampleMethod;
    }

    public static ClassOrInterfaceDeclaration getNewClassCode(String codePath, List<String> methodRemoveList) throws FileNotFoundException{
        //List<String> newClass = new ArrayList<>();
        CompilationUnit cp = StaticJavaParser.parse(new File(codePath));
        VoidVisitor<List<String>> classNameVisitor = new sameClassTest.ClassNameCollector();
        classNameVisitor.visit(cp, methodRemoveList);
        return sampleClass;
    }

    public static ClassOrInterfaceDeclaration getClassByPath(String codePath) throws FileNotFoundException{
        //List<String> newClass = new ArrayList<>();
        CompilationUnit cp = StaticJavaParser.parse(new File(codePath));
        VoidVisitor<List<String>> classNameVisitor = new sameClassTest.gatAnClassByPath();
        classNameVisitor.visit(cp, new ArrayList<>());
        return sampleClass;
    }

    public static String getDataItemPath(String projectName, String methodName, String srcClassName, String tagClassName, String posNeg, int itemPos){
        String s;
        if (posNeg == "pos"){
            s = "_item_pos_";
        }else{
            s = "_item_neg_";
        }
        File file = new File(dataItemPath+projectName+'_'+methodName+'_'+srcClassName+'_'+tagClassName +s+ itemPos);
        if (!file.exists()) {
            file.mkdirs();
        }
        return dataItemPath+projectName+'_'+methodName+'_'+srcClassName+'_'+tagClassName +s+ itemPos;
    }

    public static void writeSampleMethodToDataItem(String srcJavaSavePath, MethodDeclaration sampleMethod) throws IOException{
        //保存Java，写入本地文件
        BufferedWriter aClass = new BufferedWriter(new FileWriter(srcJavaSavePath,false));
        aClass.write(sampleMethod.toString());
        aClass.newLine();
        aClass.close();
    }
    
    public static void writeSampleClassToDataItem(String classSavePath, ClassOrInterfaceDeclaration sampleClass) throws IOException{
        //保存Java，写入本地文件
        BufferedWriter aClass = new BufferedWriter(new FileWriter(classSavePath,false));
        aClass.write(sampleClass.toString());
        aClass.newLine();
        aClass.close();
    }

    public static void appendLabelItem(String labelTxtPath, String labelInfo) throws IOException{
        BufferedWriter txt = new BufferedWriter(new FileWriter(labelTxtPath,true));
        txt.write(labelInfo.toString());
        txt.newLine();
        txt.close();
    }

  

    private static void parse_file(String csvPath) throws FileNotFoundException {

        try {  
            BufferedReader reader = new BufferedReader(new FileReader(csvPath)); 
            reader.readLine();//第一行信息，为标题信息，不用,如果需要，注释掉 
            String line = null;  
            while((line=reader.readLine())!=null){  
                String item[] = line.split(",");//CSV格式文件为逗号分隔符文件
                  
                String projectName = item[1];
                String srcClassName = item[3].split("\\/")[item[3].split("\\/").length-1].split("\\.")[0];
                String tagClassName = item[4].split("\\/")[item[4].split("\\/").length-1].split("\\.")[0];
                
                //System.out.println(projectName);
                /**
                 * 正样本：
                 * 原始的 needRefactmethodName -> TagClass  1
                 * 增强的 needRefactmethodName -> TagClass-(some 与 needRefactmethodName无关的方法)  1
                 * 增强的 needRefactmethodName -> TagClass-(some 与 needRefactmethodName无关的方法)  1
                 * 
                 * 负样本：
                 *  (each method in classA) & classB  0
                 *  (each method in classB) & classA  0
                 * }
                 */
                
                // 首先考虑正样本，得到了DateItem（a）初步的原始信息
                String srcClassPathPos = rootPath + '/' + projectName + "/a/" + srcClassName + ".java";
                String tagClassPathPos = rootPath  + '/' + projectName + "/a/" + tagClassName + ".java";

                sampleClass = getClassByPath(srcClassPathPos);
                ClassOrInterfaceDeclaration classA = sampleClass;
                sampleClass = getClassByPath(tagClassPathPos);
                ClassOrInterfaceDeclaration classB = sampleClass;
                if (classA.toString() == classB.toString()){
                    System.out.println(srcClassPathPos+"   "+tagClassPathPos);
                }
                
                //其次考虑负样本,从重构后的类对中获取，首先考虑两种Case： (each method in classA) & classB || (each method in classB) & classA 
                String srcClassPathNeg = rootPath + '/' + projectName + "/b/" + srcClassName + ".java";
                String tagClassPathNeg = rootPath  + '/' + projectName + "/b/" + tagClassName + ".java";
                sampleClass = getClassByPath(srcClassPathNeg);
                ClassOrInterfaceDeclaration classC = sampleClass;
                sampleClass = getClassByPath(tagClassPathNeg);
                ClassOrInterfaceDeclaration classD = sampleClass;
                if (classC.toString() == classD.toString()){
                    System.out.println(srcClassPathPos+"  "+tagClassPathNeg);
                }
                
            }
            reader.close();
        } catch (Exception e) {  
            e.printStackTrace(); 
            System.out.println("Some exceptions occure ~"); 
        }  
    }

    public static void main(String[] args) throws FileNotFoundException {
        
        parse_file(csvPath);
    }
}
