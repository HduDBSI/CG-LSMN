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

// import org.apache.commons.io.FileUtils;
// import java.nio.charset.StandardCharsets;


public class ComputeDistance {
    private static String currentMethodName = "";
    private static String currentMethodText = "";
    private static Boolean mInC = false;

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

    public static double getDistance(Set<String> set1, Set<String> set2) {
        if(set1.isEmpty() && set2.isEmpty())
            return 1.0;
        return 1.0 - (double)intersection(set1,set2).size()/(double)union(set1,set2).size();
    }

    public static Set<String> union(Set<String> set1, Set<String> set2) {
        Set<String> set = new HashSet<String>();
        set.addAll(set1);
        set.addAll(set2);
        return set;
    }

    public static Set<String> intersection(Set<String> set1, Set<String> set2) {
        Set<String> set = new HashSet<String>();
        set.addAll(set1);
        set.retainAll(set2);
        return set;
    }

    //收集 public attributes
    public static class PubilcMemberNameCollector extends VoidVisitorAdapter<List<String>> {
        @Override
        public void visit(ClassOrInterfaceDeclaration n, List<String> collector){
            super.visit(n, collector);
            for(final FieldDeclaration field: n.getFields()){
                // 一行声明中，可能声明了多个成员变量，所以需要对getVariables列表进行迭代
                if(field.isPublic()){ //公有的才可以被accessed
                    for(int i = 0; i < field.getVariables().size(); i++){
                        collector.add(field.getVariable(i).getNameAsString());
                    }
                }
            }
        }
    }
    //收集 all attributes
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
    public static List<String> getMemberNames(String codePath) throws FileNotFoundException{
        List<String> memberNames = new ArrayList<>();
        CompilationUnit cp = StaticJavaParser.parse(new File(codePath));
        if (mInC){
            VoidVisitor<List<String>> memberNameVisitor = new ComputeDistance.AllMemberNameCollector();
            memberNameVisitor.visit(cp, memberNames);
        }else{
            VoidVisitor<List<String>> memberNameVisitor = new ComputeDistance.PubilcMemberNameCollector();
            memberNameVisitor.visit(cp, memberNames);
        }
        return memberNames;
    }
    //收集 public methods
    private static class PublicMethodNameCollector extends VoidVisitorAdapter<List<String>> { //MethodNameCollector，抽取方法信息
        @Override
        public void visit(MethodDeclaration md, List<String> collector){
            super.visit(md, collector);
            if(md.isPublic()){
                collector.add(md.getNameAsString());
            }
        }
    }
    //收集 all methods except current method
    private static class AllNameCollector extends VoidVisitorAdapter<List<String>> { //MethodNameCollector，抽取方法信息
        @Override
        public void visit(MethodDeclaration md, List<String> collector){
            super.visit(md, collector);
            if(!md.getNameAsString().equalsIgnoreCase(currentMethodName)){
                collector.add(md.getNameAsString());
            }else{
                currentMethodText = md.toString();
            }
        }
    }
    
    public static List<String> getMethodNames(String codePath) throws FileNotFoundException{
        List<String> methodNames = new ArrayList<>();
        CompilationUnit cp = StaticJavaParser.parse(new File(codePath));
        if (mInC){
            VoidVisitor<List<String>> methodNameVisitor = new ComputeDistance.AllNameCollector();
            methodNameVisitor.visit(cp, methodNames);
        }else{
            VoidVisitor<List<String>> methodNameVisitor = new ComputeDistance.PublicMethodNameCollector();
            methodNameVisitor.visit(cp, methodNames);
        }
        return methodNames;
    }
    //收集 class 中的 Entities
    public static Set<String> getAccessEntitiesInClass(String codePath) throws FileNotFoundException{
        List<String> methodNames = getMethodNames(codePath);
        List<String> memberNames = getMemberNames(codePath);
        List<String> list = new ArrayList<>();
        list.addAll(methodNames);
        list.addAll(memberNames);
        Set<String> accessEntitiesInClass = new HashSet<String>(list);
        return accessEntitiesInClass;
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
    public static Set<String> getAccessEntitiesInMethodByCode(String codeText){
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
    
    public static void generateClassClassLiuFormatDataset(String labelPath, String rootPath, String savePath) throws FileNotFoundException{
        List<String> labelLines = readFromFile(new File(labelPath));
        List<String> txtContent = new ArrayList<>();
        for (int i = 0; i < labelLines.size(); i += 2) {
            //System.out.println(datalabel);
            String[] itemdata = labelLines.get(i).split("  ");
            String classPathX = rootPath + itemdata[0] + "/" + itemdata[1] + ".java";
            String classPathY = rootPath + itemdata[0] + "/" + itemdata[2] + ".java";
            String itemlabel = itemdata[3];
            //System.out.println(labelLines.get(i) + " " + itemdata[1]+" "+itemdata[2]+" "+itemdata[3]);

            String itemInfo = itemdata[0].split("/")[1];
            String name_m = itemdata[0].split("/")[1].split("_")[itemdata[0].split("/")[1].split("_").length-6];
            String name_ec = itemdata[1];
            String name_tc = itemdata[2];
            currentMethodName = name_m;

            // for classX, m in class, mInC = true
            mInC = true;
            Set<String> accessEntitiesInClassX = getAccessEntitiesInClass(classPathX);
            Set<String> accessEntitiesInCurrentMethod = getAccessEntitiesInMethodByCode(currentMethodText);
            // System.out.println(currentMethodName);
            // System.out.println(currentMethodText);
            double distance_m_ec =  getDistance(accessEntitiesInCurrentMethod, accessEntitiesInClassX);

            // for classY, m out class, mInC = false
            mInC = false;
            Set<String> accessEntitiesInClassY = getAccessEntitiesInClass(classPathY);
            double distance_m_tc =  getDistance(accessEntitiesInCurrentMethod, accessEntitiesInClassY);

            txtContent.add(itemInfo + " " + name_m + " " + name_ec + " " + name_tc + " " + distance_m_ec + " " + distance_m_tc + " " + itemlabel);
        }
        writeToTxt(savePath, String.join("\n",txtContent));
    }

    public static void generateMethodClassLiuFormatDataset(String labelPath, String rootPath, String savePath, String sourceCodePath) throws FileNotFoundException{
        List<String> labelLines = readFromFile(new File(labelPath));
            List<String> txtContent = new ArrayList<>();
            for (int i = 0; i < labelLines.size(); i++) {
                try {
                    //System.out.println(datalabel);
                    String[] itemdata = labelLines.get(i).split("  ");

                    String itemInfo = itemdata[0].split("/")[1];
                    String name_m = itemdata[1];
                    String name_ec = itemdata[0].split("/")[1].split("_")[itemdata[0].split("/")[1].split("_").length-5];
                    String name_tc = itemdata[2];
                    currentMethodName = name_m;
                    String classPathX = "";
                    String itemlabel = itemdata[3];
                    if (itemlabel.equalsIgnoreCase("1")){ //正样本来自/a/
                        List<String> projectName = new ArrayList<String>();
                        for (int j = 0; j < itemdata[0].split("/")[1].split("_").length-6; j++){
                            projectName.add(itemdata[0].split("/")[1].split("_")[j]);
                        }
                        classPathX = sourceCodePath + String.join("_", projectName) + "/a/" + name_ec + ".java"; //classX 和 currentMethod均来自 sourceCode
                    }else{ //负样本来自/b/
                        List<String> projectName = new ArrayList<String>();
                        for (int j = 0; j < itemdata[0].split("/")[1].split("_").length-6; j++){
                            projectName.add(itemdata[0].split("/")[1].split("_")[j]);
                        }
                        classPathX = sourceCodePath + String.join("_", projectName) + "/b/" + name_ec + ".java"; //classX 和 currentMethod均来自 sourceCode
                    }
                    String classPathY = rootPath + itemdata[0] + "/" + itemdata[2] + ".java";
                    
                    // for classX, m in class, mInC = true
                    mInC = true;
                    Set<String> accessEntitiesInClassX = getAccessEntitiesInClass(classPathX);
                    Set<String> accessEntitiesInCurrentMethod = getAccessEntitiesInMethodByCode(currentMethodText);
                    // System.out.println(currentMethodName);
                    // System.out.println(currentMethodText);
                    double distance_m_ec =  getDistance(accessEntitiesInCurrentMethod, accessEntitiesInClassX);

                    // for classY, m out class, mInC = false
                    mInC = false;
                    Set<String> accessEntitiesInClassY = getAccessEntitiesInClass(classPathY);
                    double distance_m_tc =  getDistance(accessEntitiesInCurrentMethod, accessEntitiesInClassY);

                    txtContent.add(itemInfo + " " + name_m + " " + name_ec + " " + name_tc + " " + distance_m_ec + " " + distance_m_tc + " " + itemlabel);
                } catch (Exception e) {
                    System.out.println("not found + 1");
                }
            }
            writeToTxt(savePath, String.join("\n",txtContent));
    }

    public static void main(String[] args) throws Throwable {

        Boolean CCFlag = false;
        if (CCFlag){
            String rootPath = "/home/yqx/Documents/my-FeatureEnvy-dataset/Dataset_Class_Class/";
            String labelPath = rootPath + "new_labels.txt";
            String savePath = rootPath + "dataset_liu_format.txt";
            currentMethodName = "";
            currentMethodText = "";
            mInC = false;
            //两个需求（1/2）：给到 dataItem 中每一条 label，生成对应格式的数据条目保存在txt中，
            //每个数据单元： itemInfo，name（m），name（ec），name（tc）, distance（m，ec），distance（m，tc）
            generateClassClassLiuFormatDataset(labelPath, rootPath, savePath);
        }else{
            String rootPath = "/home/yqx/Documents/my-FeatureEnvy-dataset/Dataset_Method_Class/";
            String labelPath = rootPath + "new_labels.txt";
            String savePath = rootPath + "dataset_liu_format.txt";
            String sourceCodePath = "/home/yqx/Documents/my-FeatureEnvy-dataset/SourceCode/";
            currentMethodName = "";
            currentMethodText = "";
            mInC = false;
            //两个需求（2/2）：给到 dataItem 中每一条 label，生成对应格式的数据条目保存在txt中，
            //每个数据单元： itemInfo，name（m），name（ec），name（tc）, distance（m，ec），distance（m，tc）
            //m 来自 sourceCode 中
            //note: 正样本来自/a/ | 负样本来自/b/
            generateMethodClassLiuFormatDataset(labelPath, rootPath, savePath, sourceCodePath);
        }     
    } 
}
