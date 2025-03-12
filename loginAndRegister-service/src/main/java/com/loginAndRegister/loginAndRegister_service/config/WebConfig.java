package com.loginAndRegister.loginAndRegister_service.config;

import org.springframework.context.annotation.Configuration;
import org.springframework.web.servlet.config.annotation.CorsRegistry;
import org.springframework.web.servlet.config.annotation.WebMvcConfigurer;

@Configuration
public class WebConfig implements WebMvcConfigurer {
    @Override
    public void addCorsMappings(CorsRegistry registry) {
        registry.addMapping("/**")
                .allowedOrigins("http://localhost:8800") // ✅ Match frontend origin
                .allowedMethods("GET", "POST", "PUT", "DELETE", "OPTIONS")
                .allowCredentials(true) // ✅ Crucial for cookies!
                .allowedHeaders("*")
                .exposedHeaders("Set-Cookie"); // ✅ Allow browser to see `Set-Cookie`
    }
}
