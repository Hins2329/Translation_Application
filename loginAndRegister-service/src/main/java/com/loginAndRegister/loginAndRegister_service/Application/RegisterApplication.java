package com.loginAndRegister.loginAndRegister_service.Application;

import com.loginAndRegister.loginAndRegister_service.Domain.model.User;
import com.loginAndRegister.loginAndRegister_service.Service.UserService;
import org.springframework.context.annotation.ComponentScan;
import org.springframework.stereotype.Component;

@Component
@ComponentScan("com.loginAndRegister.loginAndRegister_service.infrastructure")
public class RegisterApplication {
    private final UserService userService;

    public RegisterApplication(UserService userService) {
        this.userService = userService;
    }


    public void register(User user){
        userService.registerUser(user);

    }

}
