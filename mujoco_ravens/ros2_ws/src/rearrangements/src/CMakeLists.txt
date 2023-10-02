install(
	PROGRAMS
	scripts/hard_coded_stack.py
	DESTINATION lib/${PROJECT_NAME}
	)

install(
	DIRECTORY config launch scripts
	DESTINATION share/${PROJECT_NAME}
)
