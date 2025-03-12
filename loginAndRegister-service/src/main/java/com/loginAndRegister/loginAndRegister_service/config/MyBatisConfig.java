//package com.loginAndRegister.loginAndRegister_service.config;
//
//import org.apache.ibatis.datasource.pooled.PooledDataSource;
//import org.apache.ibatis.session.SqlSessionFactory;
//import org.mybatis.spring.SqlSessionFactoryBean;
//import org.springframework.beans.factory.annotation.Value;
//import org.springframework.context.annotation.Bean;
//import org.springframework.context.annotation.Configuration;
//import org.springframework.core.io.support.PathMatchingResourcePatternResolver;
//
//import javax.sql.DataSource;
//
//@Configuration
//public class MyBatisConfig {
//
//    @Value("${spring.datasource.url}")
//    private String url;
//
//    @Value("${spring.datasource.username}")
//    private String username;
//
//    @Value("${spring.datasource.password}")
//    private String password;
//
//    @Value("${spring.datasource.driver-class-name}")
//    private String driverClassName;
//
//    public DataSource dataSource(){
//        PooledDataSource dataSource = new PooledDataSource();
//        dataSource.setDriver(driverClassName);
//        dataSource.setUrl(url);
//        dataSource.setUsername(username);
//        dataSource.setPassword(password);
//        return dataSource;
//    }
//
//    @Bean
//    public SqlSessionFactory sqlSessionFactory() throws Exception {
//        SqlSessionFactoryBean sessionFactoryBean = new SqlSessionFactoryBean();
//        sessionFactoryBean.setDataSource(dataSource());
//        sessionFactoryBean.setMapperLocations(
//                new PathMatchingResourcePatternResolver().getResources("classpath:mapper/*.xml"));
//        return sessionFactoryBean.getObject();
//    }
//
//
//}
//
//
//
//
//-----------------
package com.loginAndRegister.loginAndRegister_service.config;

import org.apache.ibatis.session.SqlSessionFactory;
import org.mybatis.spring.SqlSessionFactoryBean;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.core.io.support.PathMatchingResourcePatternResolver;
import org.apache.ibatis.datasource.pooled.PooledDataSource;
import javax.sql.DataSource;

@Configuration
public class MyBatisConfig {

    // Using @Value to inject properties from application.properties
    @Value("${spring.datasource.url}")
    private String url;

    @Value("${spring.datasource.username}")
    private String username;

    @Value("${spring.datasource.password}")
    private String password;

    @Value("${spring.datasource.driver-class-name}")
    private String driverClassName;

    // Custom DataSource setup, can be simplified using Spring Boot's auto-configuration
    @Bean
    public DataSource dataSource() {
        PooledDataSource dataSource = new PooledDataSource();
        dataSource.setDriver(driverClassName);
        dataSource.setUrl(url);
        dataSource.setUsername(username);
        dataSource.setPassword(password);
        return dataSource;
    }

    // SqlSessionFactory bean setup, make sure to use the correct locations for your mapper XML files
    @Bean
    public SqlSessionFactory sqlSessionFactory() throws Exception {
        SqlSessionFactoryBean sessionFactoryBean = new SqlSessionFactoryBean();
        sessionFactoryBean.setDataSource(dataSource());  // Use the custom DataSource

        // Setting mapper locations
//        sessionFactoryBean.setMapperLocations(
//                new PathMatchingResourcePatternResolver().getResources("classpath:mapper/*.xml")
//        );
        return sessionFactoryBean.getObject();
    }
}