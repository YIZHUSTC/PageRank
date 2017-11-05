import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.chain.ChainMapper;
import org.apache.hadoop.mapreduce.lib.input.MultipleInputs;
import org.apache.hadoop.mapreduce.lib.input.TextInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

public class  UnitMultiplication {

    public static class TransitionMapper extends Mapper<Object, Text, Text, Text> {

        @Override
        public void map(Object key, Text value, Context context) throws IOException, InterruptedException {

            //input format: fromPage\t toPage1,toPage2,toPage3
            //target: build transition matrix unit -> fromPage\t toPage=probability

            String line = value.toString().toLowerCase().trim();
            String[] fromTo = line.split("\t");

            if(fromTo.length == 0 || fromTo[0].trim().length() == 0 || fromTo[1].trim().length() == 0) {
                Logger logger = LoggerFactory.getLogger(TransitionMapper.class);
                logger.info("Not a valid fromTo line.");
                return;
            }

            String[] tos = fromTo[1].split(",");
            for (String to : tos) {
                context.write(new Text(fromTo[0]), new Text(to + "=" + (double) 1 / tos.length));
            }
        }
    }

    public static class PRMapper extends Mapper<Object, Text, Text, Text> {

        @Override
        public void map(Object key, Text value, Context context) throws IOException, InterruptedException {

            //input format: Page\t PageRank
            //target: write to reducer

            String line = value.toString().toLowerCase().trim();
            String[] pr = line.split("\t");
            context.write(new Text(pr[0]), new Text(pr[1]));
        }
    }

    public static class MultiplicationReducer extends Reducer<Text, Text, Text, Text> {

        float beta;

        @Override
        public void setup(Context context) {
            Configuration conf = context.getConfiguration();
            beta = conf.getFloat("beta", 0.2f);
        }

        @Override
        public void reduce(Text key, Iterable<Text> values, Context context)
                throws IOException, InterruptedException {

            //input key = fromPage value=<toPage=probability..., pageRankUnit>
            //target: get the unit multiplication

            List<String> toPage = new ArrayList<String>();
            List<Double> probability = new ArrayList<Double>();
            double pageRankUnit = 0;
            for (Text value : values) {
                if (value.toString().contains("=")) {
                     toPage.add(value.toString().split("=")[0]);
                     probability.add(Double.parseDouble(value.toString().split("=")[1]));
                } else {
                    pageRankUnit = Double.parseDouble(value.toString());
                }
            }

            for(int i = 0; i < toPage.size(); i++) {
                //transition matrix * pageRank matrix * (1-beta)
                context.write(new Text(toPage.get(i)), new Text(String.valueOf(probability.get(i) * pageRankUnit * (1 - beta))));
            }

        }
    }

    public static void main(String[] args) throws Exception {

        Configuration conf = new Configuration();
        Job job = Job.getInstance(conf);
        job.setJarByClass(UnitMultiplication.class);

        conf.setFloat("beta", Float.parseFloat(args[3]));

        //how chain two mapper classes?
        ChainMapper.addMapper(job, TransitionMapper.class, Object.class, Text.class, Text.class, Text.class, conf);
        ChainMapper.addMapper(job, PRMapper.class, Object.class, Text.class, Text.class, Text.class, conf);
        job.setReducerClass(MultiplicationReducer.class);

        job.setOutputKeyClass(Text.class);
        job.setOutputValueClass(Text.class);

        MultipleInputs.addInputPath(job, new Path(args[0]), TextInputFormat.class, TransitionMapper.class);
        MultipleInputs.addInputPath(job, new Path(args[1]), TextInputFormat.class, PRMapper.class);

        FileOutputFormat.setOutputPath(job, new Path(args[2]));
        job.waitForCompletion(true);
    }

}
